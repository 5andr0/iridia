// Headless Direct3D 12 WARP run of one compute program over three read-write buffers, u0 and u2 raw and u1 structured
// of stride 8 (space 0): each starts with the words `seed` gives, one group is dispatched, and each buffer's words after
// are printed as a JSON object (buffer_lanes16.warp.json is its output for buffer_lanes16.dxil).
//
//   cl /nologo /std:c++17 /EHsc /O2 buffer_lanes16.warp.cpp /Fe:lanes-warp.exe    (a Visual Studio x64 prompt)
//   lanes-warp buffer_lanes16.dxil > buffer_lanes16.warp.json
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <cstdint>
#include <cstdio>
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

// Word k of buffer b: two halves in [1, 2) each (no NaN, infinity or denormal), different in every buffer and word.
static uint32_t seed(unsigned b, unsigned k) {
    auto half = [](unsigned j) { return 0x3c00u + (j * 37u) % 0x0400u; };
    unsigned j = (b * 1000u + k) * 2u;
    return half(j) | (half(j + 1u) << 16);
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: lanes-warp <program.dxil>");
        std::ifstream f(argv[1], std::ios::binary);
        if (!f) throw std::runtime_error(argv[1]);
        std::vector<char> code{std::istreambuf_iterator<char>(f), {}};
        ComPtr<ID3D12Debug> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) debug->EnableDebugLayer();
        ComPtr<IDXGIFactory4> factory;
        check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "factory");
        ComPtr<IDXGIAdapter> adapter;
        check(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)), "WARP adapter");
        ComPtr<ID3D12Device> dev;
        check(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev)), "device");
        D3D12_FEATURE_DATA_D3D12_OPTIONS4 o4 = {};
        check(dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &o4, sizeof o4), "options4");
        if (!o4.Native16BitShaderOpsSupported) throw std::runtime_error("no native 16-bit shader operations");
        D3D12_COMMAND_QUEUE_DESC qd = {};
        ComPtr<ID3D12CommandQueue> queue;
        check(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)), "queue");
        ComPtr<ID3D12CommandAllocator> allocator;
        check(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "allocator");
        ComPtr<ID3D12GraphicsCommandList> list;
        check(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list)),
              "list");

        D3D12_DESCRIPTOR_RANGE range = {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 3, 0, 0, 0};
        D3D12_ROOT_PARAMETER param = {};
        param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        param.DescriptorTable = {1, &range};
        D3D12_ROOT_SIGNATURE_DESC rd = {1, &param, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        ComPtr<ID3DBlob> blob, errors;
        check(D3D12SerializeRootSignature(&rd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errors), "root serialize");
        ComPtr<ID3D12RootSignature> root;
        check(dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)), "root");
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = root.Get();
        pd.CS = {code.data(), code.size()};
        ComPtr<ID3D12PipelineState> pso;
        check(dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso)), "pso");

        D3D12_DESCRIPTOR_HEAP_DESC hd = {D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 3,
                                         D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        ComPtr<ID3D12DescriptorHeap> heap;
        check(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)), "heap");
        UINT step = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        const unsigned words[3] = {128, 16, 64};
        ComPtr<ID3D12Resource> gpu[3], upload[3], readback[3];
        auto desc = [](UINT64 n, bool uav) {
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
        for (unsigned b = 0; b < 3; b++) {
            UINT64 bytes = words[b] * 4ull;
            gpu[b] = make(desc(bytes, true), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
            upload[b] = make(desc(bytes, false), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
            readback[b] = make(desc(bytes, false), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
            uint32_t* p;
            check(upload[b]->Map(0, nullptr, reinterpret_cast<void**>(&p)), "map");
            for (unsigned k = 0; k < words[b]; k++) p[k] = seed(b, k);
            upload[b]->Unmap(0, nullptr);
            list->CopyBufferRegion(gpu[b].Get(), 0, upload[b].Get(), 0, bytes);
            D3D12_RESOURCE_BARRIER br = {};
            br.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            br.Transition = {gpu[b].Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COPY_DEST,
                             D3D12_RESOURCE_STATE_UNORDERED_ACCESS};
            list->ResourceBarrier(1, &br);
            D3D12_UNORDERED_ACCESS_VIEW_DESC u = {};
            u.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            if (b == 1) {
                u.Format = DXGI_FORMAT_UNKNOWN;
                u.Buffer.NumElements = words[b] / 2;
                u.Buffer.StructureByteStride = 8;
            } else {
                u.Format = DXGI_FORMAT_R32_TYPELESS;
                u.Buffer.NumElements = words[b];
                u.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            }
            auto h = heap->GetCPUDescriptorHandleForHeapStart();
            h.ptr += b * step;
            dev->CreateUnorderedAccessView(gpu[b].Get(), nullptr, &u, h);
        }
        ID3D12DescriptorHeap* heaps[] = {heap.Get()};
        list->SetDescriptorHeaps(1, heaps);
        list->SetPipelineState(pso.Get());
        list->SetComputeRootSignature(root.Get());
        list->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
        list->Dispatch(1, 1, 1);
        for (unsigned b = 0; b < 3; b++) {
            D3D12_RESOURCE_BARRIER br = {};
            br.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            br.Transition = {gpu[b].Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                             D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE};
            list->ResourceBarrier(1, &br);
            list->CopyBufferRegion(readback[b].Get(), 0, gpu[b].Get(), 0, words[b] * 4ull);
        }
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
        printf("{");
        for (unsigned b = 0; b < 3; b++) {
            uint32_t* p;
            check(readback[b]->Map(0, nullptr, reinterpret_cast<void**>(&p)), "readback");
            printf("%s\"u%u\": [", b ? ", " : "", b);
            for (unsigned k = 0; k < words[b]; k++) printf("%s%u", k ? ", " : "", p[k]);
            printf("]");
            readback[b]->Unmap(0, nullptr);
        }
        printf("}\n");
        return 0;
    } catch (const std::exception& ex) {
        fprintf(stderr, "FAIL %s\n", ex.what());
        return 1;
    }
}
