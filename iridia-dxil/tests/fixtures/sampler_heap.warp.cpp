// Headless Direct3D 12 WARP draw of sampler_heap.dxil (ps_6_6) into one pixel: the pixel program samples its bound texture
// (t0, one colour) with the sampler `SamplerDescriptorHeap[Indices.sampler_index]` (Indices at b0, space 0), the root
// signature lets it index the sampler heap directly, and the heap holds 48 samplers, all point filtering with clamp. The
// vertex program (sampler_heap.vs.dxil) reads its outputs from `cbuffer Case : register(b0, space1) { uint4 r[32]; }`: the
// pixel program's TEXCOORD0 is row 1. Three draws, one per sampler index; each prints one JSON object of the pixel's
// four words (sampler_heap.warp.json is the output for sampler_heap.dxil).
//
//   cl /nologo /std:c++17 /EHsc /O2 sampler_heap.warp.cpp /Fe:sampler-heap-warp.exe    (a Visual Studio x64 prompt)
//   sampler-heap-warp sampler_heap.dxil sampler_heap.vs.dxil > sampler_heap.warp.json
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
using Microsoft::WRL::ComPtr;

static void check(HRESULT hr, const char* what) {
    if (FAILED(hr)) {
        char b[256];
        sprintf_s(b, "%s: 0x%08x", what, unsigned(hr));
        throw std::runtime_error(b);
    }
}

static std::vector<char> read(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error(path);
    return {std::istreambuf_iterator<char>(f), {}};
}

static uint32_t bits(float x) {
    uint32_t w;
    memcpy(&w, &x, 4);
    return w;
}

static const unsigned SAMPLERS = 48;

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: sampler-heap-warp <ps.dxil> <vs.dxil>");
        std::vector<char> ps = read(argv[1]), vs = read(argv[2]);
        ComPtr<IDXGIFactory4> factory;
        check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "factory");
        ComPtr<IDXGIAdapter> adapter;
        check(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)), "WARP adapter");
        ComPtr<ID3D12Device> dev;
        check(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev)), "device");
        D3D12_FEATURE_DATA_SHADER_MODEL sm = {D3D_SHADER_MODEL_6_6};
        check(dev->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof sm), "shader model");
        if (sm.HighestShaderModel < D3D_SHADER_MODEL_6_6) throw std::runtime_error("no shader model 6.6");
        D3D12_COMMAND_QUEUE_DESC qd = {};
        ComPtr<ID3D12CommandQueue> queue;
        check(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)), "queue");
        ComPtr<ID3D12CommandAllocator> allocator;
        check(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "allocator");
        ComPtr<ID3D12GraphicsCommandList> list;
        check(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list)),
              "list");

        // b0 space 0 (the pixel program's Indices) and b0 space 1 (the vertex program's Case) as root constant buffer
        // views, t0 as a table into the heap; the sampler heap is indexed directly
        D3D12_DESCRIPTOR_RANGE srv = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
        D3D12_ROOT_PARAMETER params[3] = {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        params[0].Descriptor = {0, 0};
        params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        params[1].Descriptor = {0, 1};
        params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
        params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[2].DescriptorTable = {1, &srv};
        params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        D3D12_ROOT_SIGNATURE_DESC rd = {3, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED};
        ComPtr<ID3DBlob> blob, errors;
        check(D3D12SerializeRootSignature(&rd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errors), "root serialize");
        ComPtr<ID3D12RootSignature> root;
        check(dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)), "root");

        D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = root.Get();
        pd.VS = {vs.data(), vs.size()};
        pd.PS = {ps.data(), ps.size()};
        pd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        pd.SampleMask = UINT_MAX;
        pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pd.RasterizerState.FrontCounterClockwise = FALSE;
        pd.RasterizerState.DepthClipEnable = TRUE;
        pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pd.NumRenderTargets = 1;
        pd.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
        pd.SampleDesc.Count = 1;
        ComPtr<ID3D12PipelineState> pso;
        check(dev->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&pso)), "pso");

        auto heap_props = [](D3D12_HEAP_TYPE t) {
            D3D12_HEAP_PROPERTIES hp = {};
            hp.Type = t;
            return hp;
        };
        auto buffer_desc = [](UINT64 n) {
            D3D12_RESOURCE_DESC d = {};
            d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            d.Width = n;
            d.Height = 1;
            d.DepthOrArraySize = 1;
            d.MipLevels = 1;
            d.SampleDesc.Count = 1;
            d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            return d;
        };
        auto texture_desc = [](UINT w, UINT h, D3D12_RESOURCE_FLAGS flags) {
            D3D12_RESOURCE_DESC d = {};
            d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            d.Width = w;
            d.Height = h;
            d.DepthOrArraySize = 1;
            d.MipLevels = 1;
            d.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
            d.SampleDesc.Count = 1;
            d.Flags = flags;
            return d;
        };
        auto make = [&](D3D12_RESOURCE_DESC d, D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES state,
                        const D3D12_CLEAR_VALUE* clear = nullptr) {
            ComPtr<ID3D12Resource> r;
            auto hp = heap_props(type);
            check(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, state, clear, IID_PPV_ARGS(&r)), "resource");
            return r;
        };
        auto barrier = [&](ID3D12Resource* r, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
            D3D12_RESOURCE_BARRIER br = {};
            br.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            br.Transition = {r, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, from, to};
            list->ResourceBarrier(1, &br);
        };

        // the texture t0: 4 by 4 texels of one colour, uploaded in the first command list
        const float colour[4] = {0.25f, -2.0f, 8.0f, 0.5f};
        ComPtr<ID3D12Resource> texture = make(texture_desc(4, 4, D3D12_RESOURCE_FLAG_NONE), D3D12_HEAP_TYPE_DEFAULT,
                                              D3D12_RESOURCE_STATE_COPY_DEST);
        const UINT pitch = 256;
        ComPtr<ID3D12Resource> texels = make(buffer_desc(pitch * 4), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        {
            uint8_t* p;
            check(texels->Map(0, nullptr, reinterpret_cast<void**>(&p)), "map texels");
            memset(p, 0, pitch * 4);
            for (unsigned y = 0; y < 4; y++)
                for (unsigned x = 0; x < 4; x++) memcpy(p + y * pitch + x * 16, colour, 16);
            texels->Unmap(0, nullptr);
        }
        D3D12_TEXTURE_COPY_LOCATION dst = {}, src = {};
        dst.pResource = texture.Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.pResource = texels.Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint.Footprint = {DXGI_FORMAT_R32G32B32A32_FLOAT, 4, 4, 1, pitch};
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        barrier(texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

        // the heaps: one shader resource view; 48 samplers
        D3D12_DESCRIPTOR_HEAP_DESC hd = {D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        ComPtr<ID3D12DescriptorHeap> heap, samplers, rtv_heap;
        check(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)), "heap");
        D3D12_SHADER_RESOURCE_VIEW_DESC tv = {};
        tv.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        tv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        tv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        tv.Texture2D.MipLevels = 1;
        dev->CreateShaderResourceView(texture.Get(), &tv, heap->GetCPUDescriptorHandleForHeapStart());
        D3D12_DESCRIPTOR_HEAP_DESC shd = {D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, SAMPLERS, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        check(dev->CreateDescriptorHeap(&shd, IID_PPV_ARGS(&samplers)), "sampler heap");
        const UINT sstep = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
        for (unsigned i = 0; i < SAMPLERS; i++) {
            D3D12_SAMPLER_DESC sd = {};
            sd.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
            sd.AddressU = sd.AddressV = sd.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
            sd.MaxLOD = D3D12_FLOAT32_MAX;
            sd.MaxAnisotropy = 1;
            auto h = samplers->GetCPUDescriptorHandleForHeapStart();
            h.ptr += SIZE_T(i) * sstep;
            dev->CreateSampler(&sd, h);
        }
        D3D12_DESCRIPTOR_HEAP_DESC rdh = {D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        check(dev->CreateDescriptorHeap(&rdh, IID_PPV_ARGS(&rtv_heap)), "rtv heap");

        // the constant buffers: Indices at 0, Case at 256; the target and its read-back
        ComPtr<ID3D12Resource> cb = make(buffer_desc(512), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        D3D12_CLEAR_VALUE clear = {};
        clear.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        ComPtr<ID3D12Resource> target = make(texture_desc(1, 1, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET),
                                             D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_RENDER_TARGET, &clear);
        dev->CreateRenderTargetView(target.Get(), nullptr, rtv_heap->GetCPUDescriptorHandleForHeapStart());
        ComPtr<ID3D12Resource> readback = make(buffer_desc(pitch), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);

        struct Case {
            const char* name;
            unsigned index;
            float uv[2];
        };
        const Case cases[] = {{"heap_first", 0, {0.5f, 0.5f}}, {"heap_middle", 17, {0.1f, 0.9f}}, {"heap_last", 47, {0.75f, 0.25f}}};
        ComPtr<ID3D12Fence> fence;
        check(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "fence");
        HANDLE e = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        UINT64 signalled = 0;
        auto submit = [&]() {
            check(list->Close(), "close");
            ID3D12CommandList* lists[] = {list.Get()};
            queue->ExecuteCommandLists(1, lists);
            check(queue->Signal(fence.Get(), ++signalled), "signal");
            check(fence->SetEventOnCompletion(signalled, e), "fence event");
            if (WaitForSingleObject(e, 30000) != WAIT_OBJECT_0) throw std::runtime_error("fence timeout");
            check(dev->GetDeviceRemovedReason(), "device removed");
        };
        submit();  // the texture upload
        for (const Case& c : cases) {
            {
                uint32_t* p;
                check(cb->Map(0, nullptr, reinterpret_cast<void**>(&p)), "map cb");
                memset(p, 0, 512);
                p[0] = c.index;
                p[64 + 4] = bits(c.uv[0]);  // Case row 1 at 256 + 16
                p[64 + 5] = bits(c.uv[1]);
                cb->Unmap(0, nullptr);
            }
            check(allocator->Reset(), "allocator reset");
            check(list->Reset(allocator.Get(), pso.Get()), "list reset");
            ID3D12DescriptorHeap* heaps[] = {heap.Get(), samplers.Get()};
            list->SetDescriptorHeaps(2, heaps);
            list->SetGraphicsRootSignature(root.Get());
            list->SetGraphicsRootConstantBufferView(0, cb->GetGPUVirtualAddress());
            list->SetGraphicsRootConstantBufferView(1, cb->GetGPUVirtualAddress() + 256);
            list->SetGraphicsRootDescriptorTable(2, heap->GetGPUDescriptorHandleForHeapStart());
            D3D12_VIEWPORT vp = {0, 0, 1, 1, 0, 1};
            D3D12_RECT sc = {0, 0, 1, 1};
            list->RSSetViewports(1, &vp);
            list->RSSetScissorRects(1, &sc);
            auto rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
            list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
            const float zero[4] = {0, 0, 0, 0};
            list->ClearRenderTargetView(rtv, zero, 0, nullptr);
            list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            list->DrawInstanced(3, 1, 0, 0);
            barrier(target.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION from = {}, to = {};
            from.pResource = target.Get();
            from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            to.pResource = readback.Get();
            to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            to.PlacedFootprint.Footprint = {DXGI_FORMAT_R32G32B32A32_FLOAT, 1, 1, 1, pitch};
            list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
            barrier(target.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
            submit();
            uint32_t* p;
            check(readback->Map(0, nullptr, reinterpret_cast<void**>(&p)), "readback");
            printf("{\"case\": \"%s\", \"target\": [%u, %u, %u, %u]}\n", c.name, p[0], p[1], p[2], p[3]);
            readback->Unmap(0, nullptr);
        }
        return 0;
    } catch (const std::exception& ex) {
        fprintf(stderr, "FAIL %s\n", ex.what());
        return 1;
    }
}
