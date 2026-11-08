// Headless Direct3D 12 WARP run of heap.dxil (cs_6_6): a compute program that takes its texture, its sampler, a source
// buffer and a target buffer from the descriptor heaps by indices in a constant buffer. The root signature lets the
// program index both heaps directly. The shader-visible heap holds, at the indices the constant buffer names:
//
//   3  the texture: 8 by 8 texels of R32G32B32A32_FLOAT, texel (x, y) = (x + 1, 2y + 1, 0.5, x - y)
//   5  the source: a structured buffer of 8 floats, element k = 1.5 + k
//   6  the target: a structured read-write buffer of 64 float4, starting at zero
//
// and the sampler heap two samplers, 0 and 1, both point filtering with clamp. One group of 8 by 8 threads is dispatched
// with Indices = (3, 1, 5, 6); the target's words after are printed as one JSON object (heap.warp.json is its output for
// heap.dxil; the program is also compiled with -Od, whose output is the same).
//
//   cl /nologo /std:c++17 /EHsc /O2 heap.warp.cpp /Fe:heap-warp.exe    (a Visual Studio x64 prompt)
//   heap-warp heap.dxil > heap.warp.json
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

static const unsigned TEXTURE_AT = 3, SOURCE_AT = 5, TARGET_AT = 6;
static const unsigned TEXELS = 8, SOURCE_ELEMENTS = 8, TARGET_ELEMENTS = 64;

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: heap-warp <program.dxil>");
        std::ifstream f(argv[1], std::ios::binary);
        if (!f) throw std::runtime_error(argv[1]);
        std::vector<char> code{std::istreambuf_iterator<char>(f), {}};
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

        // the one root parameter is the constant buffer at b0; the heaps are indexed directly
        D3D12_ROOT_PARAMETER param = {};
        param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        param.Descriptor = {0, 0};
        D3D12_ROOT_SIGNATURE_DESC rd = {1, &param, 0, nullptr,
                                        D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED |
                                            D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED};
        ComPtr<ID3DBlob> blob, errors;
        check(D3D12SerializeRootSignature(&rd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errors), "root serialize");
        ComPtr<ID3D12RootSignature> root;
        check(dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)), "root");
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = root.Get();
        pd.CS = {code.data(), code.size()};
        ComPtr<ID3D12PipelineState> pso;
        check(dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso)), "pso");

        auto buffer_desc = [](UINT64 n, bool uav) {
            D3D12_RESOURCE_DESC d = {};
            d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            d.Width = n;
            d.Height = 1;
            d.DepthOrArraySize = 1;
            d.MipLevels = 1;
            d.SampleDesc.Count = 1;
            d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            d.Flags = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
            return d;
        };
        auto make = [&](D3D12_RESOURCE_DESC d, D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES state) {
            ComPtr<ID3D12Resource> r;
            D3D12_HEAP_PROPERTIES hp = {};
            hp.Type = type;
            check(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, state, nullptr, IID_PPV_ARGS(&r)),
                  "resource");
            return r;
        };
        auto barrier = [&](ID3D12Resource* r, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
            D3D12_RESOURCE_BARRIER br = {};
            br.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            br.Transition = {r, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, from, to};
            list->ResourceBarrier(1, &br);
        };

        // the constant buffer: Indices = (texture, sampler, source, target)
        ComPtr<ID3D12Resource> cb = make(buffer_desc(256, false), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        {
            uint32_t* p;
            check(cb->Map(0, nullptr, reinterpret_cast<void**>(&p)), "map cb");
            memset(p, 0, 256);
            p[0] = TEXTURE_AT;
            p[1] = 1;
            p[2] = SOURCE_AT;
            p[3] = TARGET_AT;
            cb->Unmap(0, nullptr);
        }

        // the texture, uploaded through a buffer (a row pitch of 256 bytes)
        D3D12_RESOURCE_DESC td = {};
        td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        td.Width = TEXELS;
        td.Height = TEXELS;
        td.DepthOrArraySize = 1;
        td.MipLevels = 1;
        td.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        td.SampleDesc.Count = 1;
        ComPtr<ID3D12Resource> texture = make(td, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
        const UINT pitch = 256;
        ComPtr<ID3D12Resource> texels = make(buffer_desc(pitch * TEXELS, false), D3D12_HEAP_TYPE_UPLOAD,
                                             D3D12_RESOURCE_STATE_GENERIC_READ);
        {
            uint8_t* p;
            check(texels->Map(0, nullptr, reinterpret_cast<void**>(&p)), "map texels");
            memset(p, 0, pitch * TEXELS);
            for (unsigned y = 0; y < TEXELS; y++)
                for (unsigned x = 0; x < TEXELS; x++) {
                    float* t = reinterpret_cast<float*>(p + y * pitch) + x * 4;
                    t[0] = float(x + 1);
                    t[1] = float(2 * y + 1);
                    t[2] = 0.5f;
                    t[3] = float(int(x) - int(y));
                }
            texels->Unmap(0, nullptr);
        }
        D3D12_TEXTURE_COPY_LOCATION dst = {}, src = {};
        dst.pResource = texture.Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.pResource = texels.Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint.Footprint = {DXGI_FORMAT_R32G32B32A32_FLOAT, TEXELS, TEXELS, 1, pitch};
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        barrier(texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        // the source buffer
        ComPtr<ID3D12Resource> source = make(buffer_desc(SOURCE_ELEMENTS * 4, false), D3D12_HEAP_TYPE_DEFAULT,
                                             D3D12_RESOURCE_STATE_COPY_DEST);
        ComPtr<ID3D12Resource> source_up = make(buffer_desc(SOURCE_ELEMENTS * 4, false), D3D12_HEAP_TYPE_UPLOAD,
                                                D3D12_RESOURCE_STATE_GENERIC_READ);
        {
            float* p;
            check(source_up->Map(0, nullptr, reinterpret_cast<void**>(&p)), "map source");
            for (unsigned k = 0; k < SOURCE_ELEMENTS; k++) p[k] = 1.5f + float(k);
            source_up->Unmap(0, nullptr);
        }
        list->CopyBufferRegion(source.Get(), 0, source_up.Get(), 0, SOURCE_ELEMENTS * 4);
        barrier(source.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        // the target buffer, zero at the start
        const UINT64 target_bytes = TARGET_ELEMENTS * 16ull;
        ComPtr<ID3D12Resource> target = make(buffer_desc(target_bytes, true), D3D12_HEAP_TYPE_DEFAULT,
                                             D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        ComPtr<ID3D12Resource> readback = make(buffer_desc(target_bytes, false), D3D12_HEAP_TYPE_READBACK,
                                               D3D12_RESOURCE_STATE_COPY_DEST);

        // the shader-visible heap: the views at the indices the constant buffer names, null views elsewhere
        const unsigned heap_size = 8;
        D3D12_DESCRIPTOR_HEAP_DESC hd = {D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, heap_size,
                                         D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        ComPtr<ID3D12DescriptorHeap> heap;
        check(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)), "heap");
        const UINT step = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        auto at = [&](unsigned i) {
            auto h = heap->GetCPUDescriptorHandleForHeapStart();
            h.ptr += SIZE_T(i) * step;
            return h;
        };
        D3D12_SHADER_RESOURCE_VIEW_DESC tv = {};
        tv.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        tv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        tv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        tv.Texture2D.MipLevels = 1;
        for (unsigned i = 0; i < heap_size; i++) dev->CreateShaderResourceView(nullptr, &tv, at(i));
        dev->CreateShaderResourceView(texture.Get(), &tv, at(TEXTURE_AT));
        D3D12_SHADER_RESOURCE_VIEW_DESC sv = {};
        sv.Format = DXGI_FORMAT_UNKNOWN;
        sv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        sv.Buffer.NumElements = SOURCE_ELEMENTS;
        sv.Buffer.StructureByteStride = 4;
        dev->CreateShaderResourceView(source.Get(), &sv, at(SOURCE_AT));
        D3D12_UNORDERED_ACCESS_VIEW_DESC uv = {};
        uv.Format = DXGI_FORMAT_UNKNOWN;
        uv.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uv.Buffer.NumElements = TARGET_ELEMENTS;
        uv.Buffer.StructureByteStride = 16;
        dev->CreateUnorderedAccessView(target.Get(), nullptr, &uv, at(TARGET_AT));

        // the sampler heap: two samplers, both point filtering with clamp
        D3D12_DESCRIPTOR_HEAP_DESC shd = {D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 2, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                                          0};
        ComPtr<ID3D12DescriptorHeap> samplers;
        check(dev->CreateDescriptorHeap(&shd, IID_PPV_ARGS(&samplers)), "sampler heap");
        const UINT sstep = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
        for (unsigned i = 0; i < 2; i++) {
            D3D12_SAMPLER_DESC sd = {};
            sd.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
            sd.AddressU = sd.AddressV = sd.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
            sd.MaxLOD = D3D12_FLOAT32_MAX;
            sd.MaxAnisotropy = 1;
            auto h = samplers->GetCPUDescriptorHandleForHeapStart();
            h.ptr += SIZE_T(i) * sstep;
            dev->CreateSampler(&sd, h);
        }

        ID3D12DescriptorHeap* heaps[] = {heap.Get(), samplers.Get()};
        list->SetDescriptorHeaps(2, heaps);
        list->SetPipelineState(pso.Get());
        list->SetComputeRootSignature(root.Get());
        list->SetComputeRootConstantBufferView(0, cb->GetGPUVirtualAddress());
        list->Dispatch(1, 1, 1);
        barrier(target.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyBufferRegion(readback.Get(), 0, target.Get(), 0, target_bytes);
        check(list->Close(), "close");
        ID3D12CommandList* lists[] = {list.Get()};
        queue->ExecuteCommandLists(1, lists);
        ComPtr<ID3D12Fence> fence;
        check(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "fence");
        check(queue->Signal(fence.Get(), 1), "signal");
        HANDLE e = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        check(fence->SetEventOnCompletion(1, e), "fence event");
        if (WaitForSingleObject(e, 30000) != WAIT_OBJECT_0) throw std::runtime_error("fence timeout");
        check(dev->GetDeviceRemovedReason(), "device removed");
        uint32_t* p;
        check(readback->Map(0, nullptr, reinterpret_cast<void**>(&p)), "readback");
        printf("{\"target\": [");
        for (unsigned k = 0; k < TARGET_ELEMENTS * 4; k++) printf("%s%u", k ? ", " : "", p[k]);
        printf("]}\n");
        readback->Unmap(0, nullptr);
        return 0;
    } catch (const std::exception& ex) {
        fprintf(stderr, "FAIL %s\n", ex.what());
        return 1;
    }
}
