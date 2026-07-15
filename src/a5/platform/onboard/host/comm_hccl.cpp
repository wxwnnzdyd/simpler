/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
/**
 * HCCL backend for the comm_* distributed communication API.
 *
 * Implements the five functions declared in host/comm.h using Ascend
 * HCCL (bundled with CANN) for the bootstrap / barrier / teardown plane
 * and the public ACL IPC primitives (aclrtIpcMem* + EnablePeerAccess)
 * for the per-rank symmetric window pool (Path D).
 *
 * Scope: L3 single-host multi-card only. aclrtIpcMem* is host-local, so
 * cross-host (L4) deployments need a different windows backend -- see
 * .docs/28.l3-comm/ext.01.pr-774-review.md F2 / 05.plan.zh.md for the
 * channel-API direction.
 */

#include "platform_comm/comm.h"
#include "platform_comm/comm_context.h"

#include "common/unified_log.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <vector>
#include <unistd.h>

#include "acl/acl.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_types.h"
#include "pto/comm/async/urma/urma_channel_helper.hpp"
#include "pto/comm/async/urma/urma_hccl_defs.hpp"
#include "pto/comm/async/urma/urma_types.hpp"

// Thin wrappers around the HCCL public APIs we use. Kept as a translation
// layer in case we need to swap (e.g., InitConfig variant) later.
static inline HcclResult hccl_get_root_info(HcclRootInfo *ri) { return HcclGetRootInfo(ri); }
static inline HcclResult hccl_comm_init_root_info(uint32_t n, const HcclRootInfo *ri, uint32_t r, HcclComm *c) {
    return HcclCommInitRootInfo(n, ri, r, c);
}
static inline HcclResult hccl_barrier(HcclComm c, aclrtStream s) { return HcclBarrier(c, s); }
static inline HcclResult hccl_comm_destroy(HcclComm c) { return HcclCommDestroy(c); }

namespace {

class A5UrmaWorkspaceManager {
public:
    A5UrmaWorkspaceManager() = default;
    ~A5UrmaWorkspaceManager() { Finalize(); }

    A5UrmaWorkspaceManager(const A5UrmaWorkspaceManager &) = delete;
    A5UrmaWorkspaceManager &operator=(const A5UrmaWorkspaceManager &) = delete;

    bool Init(HcclComm comm, uint32_t rank_id, uint32_t rank_count, void *symmetric_addr, uint64_t symmetric_size) {
        comm_ = comm;
        rank_id_ = rank_id;
        rank_count_ = rank_count;
        symmetric_addr_ = symmetric_addr;
        symmetric_size_ = symmetric_size;

        if (!RegisterMemory()) {
            Finalize();
            return false;
        }
        if (!BuildChannels()) {
            Finalize();
            return false;
        }
        if (!ExtractAndFillUrmaInfo()) {
            Finalize();
            return false;
        }

        initialized_ = true;
        return true;
    }

    void Finalize() {
        FreeDeviceAddr(urma_info_device_);
        FreeDeviceAddr(eid_device_);
        channel_handles_.clear();
        hccl_channel_acquire_ = nullptr;
        if (hccl_symbol_lib_handle_ != nullptr) {
            dlclose(hccl_symbol_lib_handle_);
            hccl_symbol_lib_handle_ = nullptr;
        }
        initialized_ = false;
    }

    void *GetWorkspaceAddr() const { return urma_info_device_; }

private:
    using HcclChannelAcquireFn =
        HcclResult (*)(HcclComm, CommEngine, const HcclChannelDesc *, uint32_t, ChannelHandle *);

    struct RoceSqContextInfo {
        uint64_t sqVa;
        uint64_t headAddr;
        uint64_t tailAddr;
        uint64_t dbVa;
        uint32_t qpn;
        uint32_t wqeSize;
        uint32_t depth;
        int8_t dbMode;
        uint8_t sl;
    };

    struct RoceCqContextInfo {
        uint64_t cqVa;
        uint64_t headAddr;
        uint64_t tailAddr;
        uint64_t dbVa;
        uint32_t cqn;
        uint32_t cqeSize;
        uint32_t cqDepth;
        int8_t dbMode;
    };

    bool RegisterMemory() {
        CommMem mem{};
        mem.type = COMM_MEM_TYPE_DEVICE;
        mem.addr = symmetric_addr_;
        mem.size = symmetric_size_;

        HcclResult ret = HcclCommMemReg(comm_, kUrmaSymMemTag, &mem, &mem_handle_);
        if (ret != HCCL_SUCCESS) {
            LOG_WARN("[comm rank %u] URMA: HcclCommMemReg failed: %d", rank_id_, static_cast<int>(ret));
            return false;
        }
        return true;
    }

    bool BuildChannels() {
        std::vector<HcclChannelDesc> descs;
        descs.reserve(rank_count_ - 1);

        for (uint32_t peer = 0; peer < rank_count_; ++peer) {
            if (peer == rank_id_) continue;

            uint32_t net_layer = 0;
            uint32_t link_num = 0;
            CommLink *link_list = nullptr;
            HcclResult rc = HcclRankGraphGetLinks(comm_, net_layer, rank_id_, peer, &link_list, &link_num);
            if (rc != HCCL_SUCCESS) {
                LOG_WARN(
                    "[comm rank %u] URMA: HcclRankGraphGetLinks peer=%u failed: %d", rank_id_, peer,
                    static_cast<int>(rc)
                );
                return false;
            }

            bool found = false;
            for (uint32_t i = 0; i < link_num; ++i) {
                CommProtocol proto = link_list[i].linkAttr.linkProtocol;
                LOG_WARN(
                    "[comm rank %u] URMA: peer=%u link[%u/%u] protocol=%s(%d)", rank_id_, peer, i, link_num,
                    ProtocolName(proto), static_cast<int>(proto)
                );
                if (proto != kCommProtocolUbcCtp && proto != kCommProtocolUbcTp) continue;

                HcclChannelDesc desc;
                HcclChannelDescInit(&desc, 1);
                desc.remoteRank = peer;
                desc.notifyNum = 0;
                desc.channelProtocol = proto;
                desc.localEndpoint = link_list[i].srcEndpointDesc;
                desc.remoteEndpoint = link_list[i].dstEndpointDesc;
                desc.memHandles = &mem_handle_;
                desc.memHandleNum = 1;
                descs.push_back(desc);
                LOG_WARN(
                    "[comm rank %u] URMA: peer=%u selected channel protocol=%s(%d)", rank_id_, peer,
                    ProtocolName(proto), static_cast<int>(proto)
                );
                found = true;
                break;
            }
            if (!found) {
                LOG_WARN("[comm rank %u] URMA: no UBC_TP/CTP link to peer=%u", rank_id_, peer);
                return false;
            }
        }

        channel_handles_.resize(descs.size());
        HcclResult rc = HcclChannelAcquireResolved(
            comm_, COMM_ENGINE_AIV, descs.data(), static_cast<uint32_t>(descs.size()), channel_handles_.data()
        );
        if (rc != HCCL_SUCCESS) {
            LOG_WARN("[comm rank %u] URMA: HcclChannelAcquire failed: %d", rank_id_, static_cast<int>(rc));
            return false;
        }
        for (size_t i = 0; i < channel_handles_.size(); ++i) {
            LOG_WARN(
                "[comm rank %u] URMA: channel[%zu] handle=0x%llx class=%s", rank_id_, i,
                static_cast<unsigned long long>(channel_handles_[i]),
                IsLikelyA5DeviceVa(channel_handles_[i]) ? "device" : "opaque-host"
            );
        }
        return true;
    }

    HcclResult HcclChannelAcquireResolved(
        HcclComm comm, CommEngine engine, const HcclChannelDesc *descs, uint32_t channel_num, ChannelHandle *channels
    ) {
        HcclChannelAcquireFn fn = ResolveHcclChannelAcquire();
        if (fn == nullptr) {
            return HCCL_E_INTERNAL;
        }
        return fn(comm, engine, descs, channel_num, channels);
    }

    HcclChannelAcquireFn ResolveHcclChannelAcquire() {
        if (hccl_channel_acquire_ != nullptr) return hccl_channel_acquire_;

        dlerror();
        hccl_symbol_lib_handle_ = dlopen("libhccl.so", RTLD_NOW | RTLD_NOLOAD);
        if (hccl_symbol_lib_handle_ == nullptr) {
            hccl_symbol_lib_handle_ = dlopen("libhccl.so", RTLD_NOW);
        }
        if (hccl_symbol_lib_handle_ == nullptr) {
            LOG_WARN("[comm rank %u] URMA: dlopen libhccl.so failed: %s", rank_id_, dlerror());
            return nullptr;
        }

        auto *sym = dlsym(hccl_symbol_lib_handle_, "HcclChannelAcquire");
        if (sym == nullptr) {
            LOG_WARN("[comm rank %u] URMA: dlsym(libhccl.so, HcclChannelAcquire) failed: %s", rank_id_, dlerror());
            return nullptr;
        }
        hccl_channel_acquire_ = reinterpret_cast<HcclChannelAcquireFn>(sym);

        Dl_info info{};
        const char *path = "unknown";
        if (dladdr(sym, &info) != 0 && info.dli_fname != nullptr) {
            path = info.dli_fname;
        }
        LOG_WARN("[comm rank %u] URMA: HcclChannelAcquire resolved from %s", rank_id_, path);
        return hccl_channel_acquire_;
    }

    bool ExtractAndFillUrmaInfo() {
        std::vector<pto::comm::urma::UrmaWQCtx> wq_list(rank_count_);
        std::vector<pto::comm::urma::UrmaCqCtx> cq_list(rank_count_);
        std::vector<pto::comm::urma::UrmaMemInfo> mem_list(rank_count_);
        std::vector<uint8_t> eid_table(rank_count_ * pto::comm::urma::kUrmaEidBytes, 0);
        uint32_t local_token_id = 0;

        if (!ExtractPerPeerInfo(wq_list, cq_list, mem_list, eid_table, local_token_id)) return false;
        if (!AllocAndCopyEidTable(eid_table, mem_list)) return false;
        if (!BuildAndCopyUrmaInfoTable(wq_list, cq_list, mem_list, local_token_id)) return false;

        LOG_INFO_V0(
            "[comm rank %u] URMA workspace OK rank_count=%u localTokenId=0x%x", rank_id_, rank_count_, local_token_id
        );
        return true;
    }

    bool ExtractPerPeerInfo(
        std::vector<pto::comm::urma::UrmaWQCtx> &wq_list, std::vector<pto::comm::urma::UrmaCqCtx> &cq_list,
        std::vector<pto::comm::urma::UrmaMemInfo> &mem_list, std::vector<uint8_t> &eid_table, uint32_t &local_token_id
    ) {
        uint32_t channel_idx = 0;
        for (uint32_t peer = 0; peer < rank_count_; ++peer) {
            if (peer == rank_id_) {
                mem_list[peer].addr = reinterpret_cast<uint64_t>(symmetric_addr_);
                mem_list[peer].len = static_cast<uint32_t>(symmetric_size_);
                continue;
            }
            if (!ExtractSinglePeer(peer, channel_idx, wq_list, cq_list, mem_list, eid_table, local_token_id)) {
                return false;
            }
            ++channel_idx;
        }
        return true;
    }

    bool ExtractSinglePeer(
        uint32_t peer, uint32_t channel_idx, std::vector<pto::comm::urma::UrmaWQCtx> &wq_list,
        std::vector<pto::comm::urma::UrmaCqCtx> &cq_list, std::vector<pto::comm::urma::UrmaMemInfo> &mem_list,
        std::vector<uint8_t> &eid_table, uint32_t &local_token_id
    ) {
        ChannelHandle handle = channel_handles_[channel_idx];
        ChannelHandle entity_handle = ResolveDeviceChannelEntity(handle, peer);
        if (entity_handle == 0) return false;

        ChannelEntity host_entity{};
        SqContext sq{};
        CqContext cq{};
        RegedBufferEntity remote_buf{};
        RegedBufferEntity local_buf{};
        if (!TryReadChannelEntity(entity_handle, peer, host_entity, sq, cq, remote_buf, local_buf)) {
            LOG_WARN(
                "[comm rank %u] URMA: cannot read ChannelEntity for peer=%u handle=0x%llx entity=0x%llx", rank_id_,
                peer, static_cast<unsigned long long>(handle), static_cast<unsigned long long>(entity_handle)
            );
            return false;
        }

        RegedBufferEntity sym_remote_buf{};
        uint64_t sym_rma_addr = 0;
        uint32_t sym_rma_size = 0;
        if (!SelectSymmetricRemoteBuffer(handle, peer, host_entity, sym_remote_buf, sym_rma_addr, sym_rma_size)) {
            return false;
        }

        RegedBufferEntity sym_local_buf{};
        if (pto::comm::urma::UrmaChannelHelper::SelectSymmetricLocalBuffer(
                symmetric_size_, host_entity, peer, sym_local_buf
            ) &&
            sym_local_buf.type == REGED_BUFFER_RMA) {
            local_token_id = sym_local_buf.bufferInfo.rma.protectionInfo.memInfo.ub.tokenId;
        }

        if (!FillWqCtx(wq_list[peer], sq, peer) || !FillCqCtx(cq_list[peer], cq, peer) ||
            !FillMemInfo(mem_list[peer], sq, sym_remote_buf, sym_rma_addr, sym_rma_size, peer) ||
            !ValidatePeerContext(peer, wq_list[peer], cq_list[peer])) {
            return false;
        }
        LOG_WARN(
            "[comm rank %u] URMA: peer=%u wq{wqn=%u buf=0x%llx head=0x%llx tail=0x%llx db=0x%llx "
            "wqeShift=%u depth=%u tp=%u} cq{cqn=%u buf=0x%llx head=0x%llx tail=0x%llx db=0x%llx "
            "cqeShift=%u depth=%u} mem{addr=0x%llx len=%u tid=0x%x token=0x%x localToken=0x%x}",
            rank_id_, peer, wq_list[peer].wqn, static_cast<unsigned long long>(wq_list[peer].bufAddr),
            static_cast<unsigned long long>(wq_list[peer].headAddr),
            static_cast<unsigned long long>(wq_list[peer].tailAddr),
            static_cast<unsigned long long>(wq_list[peer].dbAddr), wq_list[peer].wqeShiftSize, wq_list[peer].depth,
            mem_list[peer].tpn, cq_list[peer].cqn, static_cast<unsigned long long>(cq_list[peer].bufAddr),
            static_cast<unsigned long long>(cq_list[peer].headAddr),
            static_cast<unsigned long long>(cq_list[peer].tailAddr),
            static_cast<unsigned long long>(cq_list[peer].dbAddr), cq_list[peer].cqeShiftSize, cq_list[peer].depth,
            static_cast<unsigned long long>(mem_list[peer].addr), mem_list[peer].len, mem_list[peer].tid,
            mem_list[peer].rmtTokenValue, local_token_id
        );

        (void)memcpy_s(
            &eid_table[peer * pto::comm::urma::kUrmaEidBytes], pto::comm::urma::kUrmaEidBytes,
            sq.contextInfo.ubJfs.remoteEID, pto::comm::urma::kUrmaEidBytes
        );
        return true;
    }

    bool TryReadChannelEntity(
        ChannelHandle entity_handle, uint32_t peer, ChannelEntity &host_entity, SqContext &sq, CqContext &cq,
        RegedBufferEntity &remote_buf, RegedBufferEntity &local_buf
    ) {
        void *dev_entity = reinterpret_cast<void *>(static_cast<uintptr_t>(entity_handle));
        LOG_WARN("[comm rank %u] URMA: read ChannelEntity begin peer=%u entity=%p", rank_id_, peer, dev_entity);
        aclError err = aclrtMemcpy(
            &host_entity, sizeof(ChannelEntity), dev_entity, sizeof(ChannelEntity), ACL_MEMCPY_DEVICE_TO_HOST
        );
        if (err != ACL_SUCCESS) {
            LOG_WARN("[comm rank %u] URMA: aclrtMemcpy(ChannelEntity) peer=%u err=%d", rank_id_, peer, err);
            return false;
        }
        if (!pto::comm::urma::UrmaChannelHelper::IsValidChannelEntityHeader(host_entity)) {
            LOG_WARN(
                "[comm rank %u] URMA: invalid ChannelEntity peer=%u magic=0x%x engine=%d", rank_id_, peer,
                host_entity.abiHeader.magicWord, static_cast<int>(host_entity.engine)
            );
            return false;
        }
        NormalizeChannelEntityPointers(entity_handle, host_entity);
        LOG_WARN(
            "[comm rank %u] URMA: entity peer=%u protocol=%s(%d) sq=%p/%u cq=%p/%u remoteBuf=%p/%u localBuf=%p/%u",
            rank_id_, peer, ProtocolName(host_entity.protocol), static_cast<int>(host_entity.protocol),
            static_cast<void *>(host_entity.sqContextAddr), host_entity.sqNum,
            static_cast<void *>(host_entity.cqContextAddr), host_entity.cqNum,
            static_cast<void *>(host_entity.remoteBufferAddr), host_entity.remoteBufferNum,
            static_cast<void *>(host_entity.localBufferAddr), host_entity.localBufferNum
        );

        if (host_entity.sqContextAddr != nullptr && host_entity.sqNum > 0 &&
            !CopyDeviceStruct(host_entity.sqContextAddr, &sq, sizeof(SqContext), peer, "SqContext")) {
            return false;
        }
        if (host_entity.cqContextAddr != nullptr && host_entity.cqNum > 0 &&
            !CopyDeviceStruct(host_entity.cqContextAddr, &cq, sizeof(CqContext), peer, "CqContext")) {
            return false;
        }
        if (host_entity.remoteBufferAddr != nullptr && host_entity.remoteBufferNum > 0 &&
            !CopyDeviceStruct(
                host_entity.remoteBufferAddr, &remote_buf, sizeof(RegedBufferEntity), peer, "RemoteBuffer"
            )) {
            return false;
        }
        if (host_entity.localBufferAddr != nullptr && host_entity.localBufferNum > 0 &&
            !CopyDeviceStruct(
                host_entity.localBufferAddr, &local_buf, sizeof(RegedBufferEntity), peer, "LocalBuffer"
            )) {
            return false;
        }
        NormalizeChannelSubStructAddresses(entity_handle, sq, cq, remote_buf, local_buf);
        LogContextRaw(peer, sq, cq);
        BackfillQueueIndexAddresses(entity_handle, host_entity, sq, cq, peer);
        return true;
    }

    void NormalizeChannelEntityPointers(ChannelHandle entity_handle, ChannelEntity &entity) {
        const uintptr_t base = static_cast<uintptr_t>(entity_handle);
        entity.localNotifyAddr = NormalizeEntityPointer(base, entity.localNotifyAddr);
        entity.remoteNotifyAddr = NormalizeEntityPointer(base, entity.remoteNotifyAddr);
        entity.localBufferAddr = static_cast<RegedBufferEntity *>(NormalizeEntityPointer(base, entity.localBufferAddr));
        entity.remoteBufferAddr =
            static_cast<RegedBufferEntity *>(NormalizeEntityPointer(base, entity.remoteBufferAddr));
        entity.sqContextAddr = static_cast<SqContext *>(NormalizeEntityPointer(base, entity.sqContextAddr));
        entity.cqContextAddr = static_cast<CqContext *>(NormalizeEntityPointer(base, entity.cqContextAddr));
    }

    void *NormalizeEntityPointer(uintptr_t entity_base, const void *ptr) {
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        if (addr == 0 || IsLikelyA5DeviceVa(static_cast<ChannelHandle>(addr))) {
            return const_cast<void *>(ptr);
        }
        return reinterpret_cast<void *>(entity_base + addr);
    }

    void NormalizeChannelSubStructAddresses(
        ChannelHandle entity_handle, SqContext &sq, CqContext &cq, RegedBufferEntity &remote_buf,
        RegedBufferEntity &local_buf
    ) {
        const uintptr_t base = static_cast<uintptr_t>(entity_handle);
        sq.contextInfo.ubJfs.sqVa = NormalizeDeviceAddress(base, sq.contextInfo.ubJfs.sqVa);
        sq.contextInfo.ubJfs.headAddr = NormalizeDeviceAddress(base, sq.contextInfo.ubJfs.headAddr);
        sq.contextInfo.ubJfs.tailAddr = NormalizeDeviceAddress(base, sq.contextInfo.ubJfs.tailAddr);
        sq.contextInfo.ubJfs.dbVa = NormalizeDeviceAddress(base, sq.contextInfo.ubJfs.dbVa);

        cq.contextInfo.ubJfc.scqVa = NormalizeDeviceAddress(base, cq.contextInfo.ubJfc.scqVa);
        cq.contextInfo.ubJfc.headAddr = NormalizeDeviceAddress(base, cq.contextInfo.ubJfc.headAddr);
        cq.contextInfo.ubJfc.tailAddr = NormalizeDeviceAddress(base, cq.contextInfo.ubJfc.tailAddr);
        cq.contextInfo.ubJfc.dbVa = NormalizeDeviceAddress(base, cq.contextInfo.ubJfc.dbVa);

        if (remote_buf.type == REGED_BUFFER_RMA) {
            remote_buf.bufferInfo.rma.addr = NormalizeDeviceAddress(base, remote_buf.bufferInfo.rma.addr);
        }
        if (local_buf.type == REGED_BUFFER_RMA) {
            local_buf.bufferInfo.rma.addr = NormalizeDeviceAddress(base, local_buf.bufferInfo.rma.addr);
        }
    }

    void BackfillQueueIndexAddresses(
        ChannelHandle entity_handle, const ChannelEntity &entity, SqContext &sq, CqContext &cq, uint32_t peer
    ) {
        if (entity.sqNum == 0 || entity.cqNum == 0 || entity.sqContextAddr == nullptr ||
            entity.cqContextAddr == nullptr) {
            return;
        }

        const uintptr_t base = static_cast<uintptr_t>(entity_handle);
        uintptr_t offset = reinterpret_cast<uintptr_t>(entity.cqContextAddr) - base;
        offset += static_cast<uintptr_t>(entity.cqNum) * sizeof(CqContext);
        const uintptr_t sq_pi_base = base + AlignUp(offset, kAivUrmaEntityAlignSize);
        const uintptr_t sq_ci_base =
            base + AlignUp(
                       (sq_pi_base - base) + static_cast<uintptr_t>(entity.sqNum) * kQueueIndexMemUnitSize,
                       kAivUrmaEntityAlignSize
                   );
        const uintptr_t cq_pi_base =
            base + AlignUp(
                       (sq_ci_base - base) + static_cast<uintptr_t>(entity.sqNum) * kQueueIndexMemUnitSize,
                       kAivUrmaEntityAlignSize
                   );
        const uintptr_t cq_ci_base =
            base + AlignUp(
                       (cq_pi_base - base) + static_cast<uintptr_t>(entity.cqNum) * kQueueIndexMemUnitSize,
                       kAivUrmaEntityAlignSize
                   );

        bool backfilled = false;
        if (sq.contextInfo.ubJfs.headAddr == 0) {
            sq.contextInfo.ubJfs.headAddr = static_cast<uint64_t>(sq_pi_base);
            backfilled = true;
        }
        if (sq.contextInfo.ubJfs.tailAddr == 0) {
            sq.contextInfo.ubJfs.tailAddr = static_cast<uint64_t>(sq_ci_base);
            backfilled = true;
        }
        if (cq.contextInfo.ubJfc.headAddr == 0) {
            cq.contextInfo.ubJfc.headAddr = static_cast<uint64_t>(cq_pi_base);
            backfilled = true;
        }
        if (cq.contextInfo.ubJfc.tailAddr == 0) {
            cq.contextInfo.ubJfc.tailAddr = static_cast<uint64_t>(cq_ci_base);
            backfilled = true;
        }
        if (backfilled) {
            LOG_WARN(
                "[comm rank %u] URMA: peer=%u backfilled queue-index addrs "
                "sqPi=0x%llx sqCi=0x%llx cqPi=0x%llx cqCi=0x%llx",
                rank_id_, peer, static_cast<unsigned long long>(sq_pi_base),
                static_cast<unsigned long long>(sq_ci_base), static_cast<unsigned long long>(cq_pi_base),
                static_cast<unsigned long long>(cq_ci_base)
            );
        }
    }

    uint64_t NormalizeDeviceAddress(uintptr_t entity_base, uint64_t addr) {
        if (addr == 0 || IsLikelyA5DeviceVa(static_cast<ChannelHandle>(addr))) {
            return addr;
        }
        return entity_base + addr;
    }

    bool CopyDeviceStruct(const void *src, void *dst, size_t size, uint32_t peer, const char *name) {
        const uintptr_t addr = reinterpret_cast<uintptr_t>(src);
        if (!IsLikelyA5DeviceVa(static_cast<ChannelHandle>(addr))) {
            LOG_WARN(
                "[comm rank %u] URMA: %s peer=%u has non-device ptr=%p; refusing host memcpy", rank_id_, name, peer, src
            );
            return false;
        }
        aclError err = aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST);
        if (err != ACL_SUCCESS) {
            LOG_WARN("[comm rank %u] URMA: aclrtMemcpy(%s) peer=%u err=%d", rank_id_, name, peer, err);
            return false;
        }
        return true;
    }

    bool SelectSymmetricRemoteBuffer(
        ChannelHandle handle, uint32_t peer, const ChannelEntity &entity, RegedBufferEntity &selected,
        uint64_t &rma_addr, uint32_t &rma_size
    ) {
        void *sym_addr = nullptr;
        uint64_t sym_size = 0;
        if (!GetRemoteMemByTag(handle, peer, &sym_addr, &sym_size)) {
            return false;
        }
        rma_addr = reinterpret_cast<uint64_t>(sym_addr);
        rma_size = static_cast<uint32_t>(sym_size);

        if (entity.remoteBufferAddr == nullptr || entity.remoteBufferNum == 0) {
            LOG_WARN("[comm rank %u] URMA: peer=%u ChannelEntity has no remote buffers", rank_id_, peer);
            return false;
        }

        for (uint32_t i = 0; i < entity.remoteBufferNum; ++i) {
            RegedBufferEntity buf{};
            if (!pto::comm::urma::UrmaChannelHelper::ReadRegedBufferEntityAt(
                    entity.remoteBufferAddr, entity.remoteBufferNum, i, peer, buf
                )) {
                continue;
            }
            if (buf.type != REGED_BUFFER_RMA) continue;
            if (buf.bufferInfo.rma.addr != rma_addr || buf.bufferInfo.rma.size != symmetric_size_) continue;

            selected = buf;
            LOG_INFO_V0(
                "[comm rank %u] URMA: selected remote buffer peer=%u addr=0x%llx size=%llu token=0x%x", rank_id_, peer,
                static_cast<unsigned long long>(rma_addr), static_cast<unsigned long long>(sym_size),
                selected.bufferInfo.rma.protectionInfo.memInfo.ub.tokenId
            );
            return true;
        }

        LOG_WARN(
            "[comm rank %u] URMA: peer=%u no RegedBufferEntity matches %s addr=0x%llx size=%llu", rank_id_, peer,
            kUrmaSymMemTag, static_cast<unsigned long long>(rma_addr), static_cast<unsigned long long>(sym_size)
        );
        return false;
    }

    bool GetRemoteMemByTag(ChannelHandle handle, uint32_t peer, void **out_addr, uint64_t *out_size) {
        CommMem *remote_mems = nullptr;
        char **mem_tags = nullptr;
        uint32_t mem_num = 0;
        HcclResult rc = HcclChannelGetRemoteMems(comm_, handle, &mem_num, &remote_mems, &mem_tags);
        if (rc != HCCL_SUCCESS) {
            LOG_WARN(
                "[comm rank %u] URMA: HcclChannelGetRemoteMems peer=%u failed: %d", rank_id_, peer, static_cast<int>(rc)
            );
            return false;
        }

        for (uint32_t i = 0; i < mem_num; ++i) {
            const char *tag = (mem_tags != nullptr && mem_tags[i] != nullptr) ? mem_tags[i] : "";
            if (std::strcmp(tag, kUrmaSymMemTag) != 0) continue;
            *out_addr = remote_mems[i].addr;
            *out_size = remote_mems[i].size;
            return true;
        }

        LOG_WARN(
            "[comm rank %u] URMA: peer=%u tag %s not found in %u remote mems", rank_id_, peer, kUrmaSymMemTag, mem_num
        );
        return false;
    }

    ChannelHandle ResolveDeviceChannelEntity(ChannelHandle handle, uint32_t peer) {
        if (IsLikelyA5DeviceVa(handle)) return handle;
        LOG_WARN(
            "[comm rank %u] URMA: peer=%u ChannelHandle is opaque host object 0x%llx; refusing private conversion. "
            "A5 URMA requires HcclChannelAcquire to return a device ChannelEntity pointer, matching PTO-ISA.",
            rank_id_, peer, static_cast<unsigned long long>(handle)
        );
        return 0;
    }

    bool FillWqCtx(pto::comm::urma::UrmaWQCtx &wq, const SqContext &sq, uint32_t peer) {
        if (sq.type == kSqContextTypeRoce) {
            const RoceSqContextInfo &roce_sq = RoceSq(sq);
            wq.wqn = roce_sq.qpn;
            wq.bufAddr = roce_sq.sqVa;
            wq.wqeShiftSize = Log2U32(roce_sq.wqeSize);
            wq.depth = roce_sq.depth;
            wq.headAddr = roce_sq.headAddr;
            wq.tailAddr = roce_sq.tailAddr;
            wq.dbMode = ToUrmaDbMode(roce_sq.dbMode);
            wq.dbAddr = roce_sq.dbVa;
            wq.sl = roce_sq.sl;
        } else if (sq.type == kSqContextTypeUbJfs) {
            wq.wqn = sq.contextInfo.ubJfs.jfsID;
            wq.bufAddr = sq.contextInfo.ubJfs.sqVa;
            wq.wqeShiftSize = Log2U32(sq.contextInfo.ubJfs.wqeSize);
            wq.depth = sq.contextInfo.ubJfs.sqDepth;
            wq.headAddr = sq.contextInfo.ubJfs.headAddr;
            wq.tailAddr = sq.contextInfo.ubJfs.tailAddr;
            wq.dbMode = pto::comm::urma::UrmaDbMode::SW_DB;
            wq.dbAddr = sq.contextInfo.ubJfs.dbVa;
            wq.sl = 0;
        } else {
            LOG_WARN("[comm rank %u] URMA: peer=%u unsupported SqContext type=%d", rank_id_, peer, sq.type);
            return false;
        }
        return true;
    }

    bool FillCqCtx(pto::comm::urma::UrmaCqCtx &cq_ctx, const CqContext &cq, uint32_t peer) {
        if (cq.type == kCqContextTypeRoce) {
            const RoceCqContextInfo &roce_cq = RoceCq(cq);
            cq_ctx.cqn = roce_cq.cqn;
            cq_ctx.bufAddr = roce_cq.cqVa;
            cq_ctx.cqeShiftSize = Log2U32(roce_cq.cqeSize);
            cq_ctx.depth = roce_cq.cqDepth;
            cq_ctx.headAddr = roce_cq.headAddr;
            cq_ctx.tailAddr = roce_cq.tailAddr;
            cq_ctx.dbMode = ToUrmaDbMode(roce_cq.dbMode);
            cq_ctx.dbAddr = roce_cq.dbVa;
        } else if (cq.type == kCqContextTypeUbJfc) {
            cq_ctx.cqn = cq.contextInfo.ubJfc.jfcID;
            cq_ctx.bufAddr = cq.contextInfo.ubJfc.scqVa;
            cq_ctx.cqeShiftSize = Log2U32(cq.contextInfo.ubJfc.cqeSize);
            cq_ctx.depth = cq.contextInfo.ubJfc.cqDepth;
            cq_ctx.headAddr = cq.contextInfo.ubJfc.headAddr;
            cq_ctx.tailAddr = cq.contextInfo.ubJfc.tailAddr;
            cq_ctx.dbMode = pto::comm::urma::UrmaDbMode::SW_DB;
            cq_ctx.dbAddr = cq.contextInfo.ubJfc.dbVa;
        } else {
            LOG_WARN("[comm rank %u] URMA: peer=%u unsupported CqContext type=%d", rank_id_, peer, cq.type);
            return false;
        }
        return true;
    }

    bool FillMemInfo(
        pto::comm::urma::UrmaMemInfo &mem, const SqContext &sq, const RegedBufferEntity &sym_remote_buf,
        uint64_t sym_rma_addr, uint32_t sym_rma_size, uint32_t peer
    ) {
        if (sq.type != kSqContextTypeUbJfs) {
            LOG_WARN("[comm rank %u] URMA: peer=%u unsupported SqContext type=%d for UB WQE", rank_id_, peer, sq.type);
            return false;
        }
        mem.tokenValueValid = true;
        mem.rmtJettyType = 1;
        mem.targetHint = 0;
        mem.tpn = sq.contextInfo.ubJfs.tpID;
        mem.tid = sym_remote_buf.bufferInfo.rma.protectionInfo.memInfo.ub.tokenId;
        mem.rmtTokenValue = sym_remote_buf.bufferInfo.rma.protectionInfo.memInfo.ub.tokenValue;
        mem.len = sym_rma_size;
        mem.addr = sym_rma_addr;
        return true;
    }

    bool
    AllocAndCopyEidTable(const std::vector<uint8_t> &eid_table, std::vector<pto::comm::urma::UrmaMemInfo> &mem_list) {
        size_t eid_dev_size = rank_count_ * pto::comm::urma::kUrmaEidBytes;
        aclError err = aclrtMalloc(&eid_device_, eid_dev_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (err != ACL_SUCCESS) {
            LOG_WARN("[comm rank %u] URMA: aclrtMalloc(eidTable) failed: %d", rank_id_, static_cast<int>(err));
            return false;
        }
        err = aclrtMemcpy(eid_device_, eid_dev_size, eid_table.data(), eid_dev_size, ACL_MEMCPY_HOST_TO_DEVICE);
        if (err != ACL_SUCCESS) {
            LOG_WARN("[comm rank %u] URMA: aclrtMemcpy(eidTable) failed: %d", rank_id_, static_cast<int>(err));
            return false;
        }
        for (uint32_t peer = 0; peer < rank_count_; ++peer) {
            mem_list[peer].eidAddr =
                reinterpret_cast<uint64_t>(static_cast<uint8_t *>(eid_device_) + peer * pto::comm::urma::kUrmaEidBytes);
        }
        return true;
    }

    bool BuildAndCopyUrmaInfoTable(
        const std::vector<pto::comm::urma::UrmaWQCtx> &wq_list, const std::vector<pto::comm::urma::UrmaCqCtx> &cq_list,
        const std::vector<pto::comm::urma::UrmaMemInfo> &mem_list, uint32_t local_token_id
    ) {
        size_t total_size = UrmaWorkspaceBytes(rank_count_);
        aclError err = aclrtMalloc(&urma_info_device_, total_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (err != ACL_SUCCESS) {
            LOG_WARN("[comm rank %u] URMA: aclrtMalloc(urmaInfo) failed: %d", rank_id_, static_cast<int>(err));
            return false;
        }

        std::vector<uint8_t> host_buf(total_size, 0);
        FillUrmaInfoLayout(host_buf, wq_list, cq_list, mem_list, local_token_id);

        err = aclrtMemcpy(urma_info_device_, total_size, host_buf.data(), total_size, ACL_MEMCPY_HOST_TO_DEVICE);
        if (err != ACL_SUCCESS) {
            LOG_WARN("[comm rank %u] URMA: aclrtMemcpy(urmaInfo) failed: %d", rank_id_, static_cast<int>(err));
            return false;
        }
        return true;
    }

    void FillUrmaInfoLayout(
        std::vector<uint8_t> &host_buf, const std::vector<pto::comm::urma::UrmaWQCtx> &wq_list,
        const std::vector<pto::comm::urma::UrmaCqCtx> &cq_list,
        const std::vector<pto::comm::urma::UrmaMemInfo> &mem_list, uint32_t local_token_id
    ) {
        auto *info = reinterpret_cast<pto::comm::urma::UrmaInfo *>(host_buf.data());
        info->qpNum = kQpNum;
        info->localTokenId = local_token_id;
        info->rankCount = rank_count_;

        uint8_t *dev_addr = static_cast<uint8_t *>(urma_info_device_) + sizeof(pto::comm::urma::UrmaInfo);
        info->sqPtr = reinterpret_cast<uint64_t>(dev_addr);
        dev_addr += sizeof(pto::comm::urma::UrmaWQCtx) * rank_count_ * kQpNum;
        info->rqPtr = reinterpret_cast<uint64_t>(dev_addr);
        dev_addr += sizeof(pto::comm::urma::UrmaWQCtx) * rank_count_ * kQpNum;
        info->scqPtr = reinterpret_cast<uint64_t>(dev_addr);
        dev_addr += sizeof(pto::comm::urma::UrmaCqCtx) * rank_count_ * kQpNum;
        info->rcqPtr = reinterpret_cast<uint64_t>(dev_addr);
        dev_addr += sizeof(pto::comm::urma::UrmaCqCtx) * rank_count_ * kQpNum;
        info->memPtr = reinterpret_cast<uint64_t>(dev_addr);

        uint8_t *host_addr = host_buf.data() + sizeof(pto::comm::urma::UrmaInfo);
        auto *sq_arr = reinterpret_cast<pto::comm::urma::UrmaWQCtx *>(host_addr);
        host_addr += sizeof(pto::comm::urma::UrmaWQCtx) * rank_count_ * kQpNum;
        auto *rq_arr = reinterpret_cast<pto::comm::urma::UrmaWQCtx *>(host_addr);
        host_addr += sizeof(pto::comm::urma::UrmaWQCtx) * rank_count_ * kQpNum;
        auto *scq_arr = reinterpret_cast<pto::comm::urma::UrmaCqCtx *>(host_addr);
        host_addr += sizeof(pto::comm::urma::UrmaCqCtx) * rank_count_ * kQpNum;
        auto *rcq_arr = reinterpret_cast<pto::comm::urma::UrmaCqCtx *>(host_addr);
        host_addr += sizeof(pto::comm::urma::UrmaCqCtx) * rank_count_ * kQpNum;
        auto *mem_arr = reinterpret_cast<pto::comm::urma::UrmaMemInfo *>(host_addr);

        for (uint32_t rank = 0; rank < rank_count_; ++rank) {
            sq_arr[rank] = wq_list[rank];
            rq_arr[rank] = wq_list[rank];
            scq_arr[rank] = cq_list[rank];
            rcq_arr[rank] = cq_list[rank];
            mem_arr[rank] = mem_list[rank];
        }
    }

    static uint64_t UrmaWorkspaceBytes(uint32_t rank_count) {
        return sizeof(pto::comm::urma::UrmaInfo) +
               static_cast<uint64_t>(rank_count) *
                   (2ULL * sizeof(pto::comm::urma::UrmaWQCtx) * kQpNum +
                    2ULL * sizeof(pto::comm::urma::UrmaCqCtx) * kQpNum + sizeof(pto::comm::urma::UrmaMemInfo) * kQpNum);
    }

    void LogContextRaw(uint32_t peer, const SqContext &sq, const CqContext &cq) const {
        LOG_WARN(
            "[comm rank %u] URMA: peer=%u SqContext type=%s(%d) raw64=[0x%llx 0x%llx 0x%llx 0x%llx "
            "0x%llx 0x%llx 0x%llx 0x%llx]",
            rank_id_, peer, SqContextName(sq.type), sq.type,
            static_cast<unsigned long long>(RawU64(sq.contextInfo.raws, 0)),
            static_cast<unsigned long long>(RawU64(sq.contextInfo.raws, 1)),
            static_cast<unsigned long long>(RawU64(sq.contextInfo.raws, 2)),
            static_cast<unsigned long long>(RawU64(sq.contextInfo.raws, 3)),
            static_cast<unsigned long long>(RawU64(sq.contextInfo.raws, 4)),
            static_cast<unsigned long long>(RawU64(sq.contextInfo.raws, 5)),
            static_cast<unsigned long long>(RawU64(sq.contextInfo.raws, 6)),
            static_cast<unsigned long long>(RawU64(sq.contextInfo.raws, 7))
        );
        LOG_WARN(
            "[comm rank %u] URMA: peer=%u CqContext type=%s(%d) raw64=[0x%llx 0x%llx 0x%llx 0x%llx "
            "0x%llx 0x%llx 0x%llx 0x%llx]",
            rank_id_, peer, CqContextName(cq.type), cq.type,
            static_cast<unsigned long long>(RawU64(cq.contextInfo.raws, 0)),
            static_cast<unsigned long long>(RawU64(cq.contextInfo.raws, 1)),
            static_cast<unsigned long long>(RawU64(cq.contextInfo.raws, 2)),
            static_cast<unsigned long long>(RawU64(cq.contextInfo.raws, 3)),
            static_cast<unsigned long long>(RawU64(cq.contextInfo.raws, 4)),
            static_cast<unsigned long long>(RawU64(cq.contextInfo.raws, 5)),
            static_cast<unsigned long long>(RawU64(cq.contextInfo.raws, 6)),
            static_cast<unsigned long long>(RawU64(cq.contextInfo.raws, 7))
        );
    }

    bool ValidatePeerContext(
        uint32_t peer, const pto::comm::urma::UrmaWQCtx &wq, const pto::comm::urma::UrmaCqCtx &cq
    ) const {
        if (wq.bufAddr != 0 && wq.headAddr != 0 && wq.tailAddr != 0 && wq.dbAddr != 0 && wq.depth != 0 &&
            cq.bufAddr != 0 && cq.headAddr != 0 && cq.tailAddr != 0 && cq.dbAddr != 0 && cq.depth != 0) {
            return true;
        }
        LOG_WARN(
            "[comm rank %u] URMA: peer=%u invalid WQ/CQ context; refusing to launch URMA kernel "
            "wq{buf=0x%llx head=0x%llx tail=0x%llx db=0x%llx depth=%u} "
            "cq{buf=0x%llx head=0x%llx tail=0x%llx db=0x%llx depth=%u}",
            rank_id_, peer, static_cast<unsigned long long>(wq.bufAddr), static_cast<unsigned long long>(wq.headAddr),
            static_cast<unsigned long long>(wq.tailAddr), static_cast<unsigned long long>(wq.dbAddr), wq.depth,
            static_cast<unsigned long long>(cq.bufAddr), static_cast<unsigned long long>(cq.headAddr),
            static_cast<unsigned long long>(cq.tailAddr), static_cast<unsigned long long>(cq.dbAddr), cq.depth
        );
        return false;
    }

    static RoceSqContextInfo RoceSq(const SqContext &sq) {
        RoceSqContextInfo out{};
        std::memcpy(&out, sq.contextInfo.raws, sizeof(out));
        return out;
    }

    static RoceCqContextInfo RoceCq(const CqContext &cq) {
        RoceCqContextInfo out{};
        std::memcpy(&out, cq.contextInfo.raws, sizeof(out));
        return out;
    }

    static pto::comm::urma::UrmaDbMode ToUrmaDbMode(int8_t db_mode) {
        if (db_mode == static_cast<int8_t>(pto::comm::urma::UrmaDbMode::HW_DB)) {
            return pto::comm::urma::UrmaDbMode::HW_DB;
        }
        return pto::comm::urma::UrmaDbMode::SW_DB;
    }

    static uint64_t RawU64(const uint8_t *raw, size_t idx) {
        uint64_t out = 0;
        std::memcpy(&out, raw + idx * sizeof(out), sizeof(out));
        return out;
    }

    static uintptr_t AlignUp(uintptr_t value, uintptr_t alignment) {
        return ((value + alignment - 1) / alignment) * alignment;
    }

    static uint32_t Log2U32(uint32_t n) { return (n <= 1) ? 0 : __builtin_ctz(n); }

    static bool IsLikelyA5DeviceVa(ChannelHandle handle) {
        return handle >= kA5DeviceVaLowerBound && handle < kA5DeviceVaUpperBound;
    }

    static const char *ProtocolName(CommProtocol protocol) {
        switch (static_cast<int>(protocol)) {
        case 1:
            return "ROCE";
        case 4:
            return "UBC_CTP";
        case 5:
            return "UBC_TP";
        case 6:
            return "UB_MEM";
        case 7:
            return "UBOE";
        default:
            return "UNKNOWN";
        }
    }

    static const char *SqContextName(int32_t type) {
        if (type == kSqContextTypeRoce) return "ROCE";
        if (type == kSqContextTypeUbJfs) return "UB_JFS";
        return "UNKNOWN";
    }

    static const char *CqContextName(int32_t type) {
        if (type == kCqContextTypeRoce) return "ROCE";
        if (type == kCqContextTypeUbJfc) return "UB_JFC";
        return "UNKNOWN";
    }

    static void FreeDeviceAddr(void *&addr) {
        if (addr) {
            aclrtFree(addr);
            addr = nullptr;
        }
    }

    static constexpr uint32_t kQpNum = 1;
    static constexpr int32_t kSqContextTypeUbJfs = 0;
    static constexpr int32_t kSqContextTypeRoce = 1;
    static constexpr int32_t kCqContextTypeUbJfc = 0;
    static constexpr int32_t kCqContextTypeRoce = 1;
    static constexpr uintptr_t kAivUrmaEntityAlignSize = 64;
    static constexpr uintptr_t kQueueIndexMemUnitSize = sizeof(void *);
    static constexpr uint64_t kA5DeviceVaLowerBound = 0x100000000000ULL;
    static constexpr uint64_t kA5DeviceVaUpperBound = 0x200000000000ULL;
    static constexpr const char *kUrmaSymMemTag = "pto_urma_sym";
    static constexpr CommProtocol kCommProtocolUbcCtp = static_cast<CommProtocol>(4);
    static constexpr CommProtocol kCommProtocolUbcTp = static_cast<CommProtocol>(5);

    HcclComm comm_{nullptr};
    uint32_t rank_id_{0};
    uint32_t rank_count_{0};
    void *symmetric_addr_{nullptr};
    uint64_t symmetric_size_{0};
    HcclMemHandle mem_handle_{nullptr};

    std::vector<ChannelHandle> channel_handles_;
    void *urma_info_device_{nullptr};
    void *eid_device_{nullptr};

    void *hccl_symbol_lib_handle_{nullptr};
    HcclChannelAcquireFn hccl_channel_acquire_{nullptr};

    bool initialized_{false};
};

}  // namespace

// ============================================================================
// Internal state
// ============================================================================

// Per-domain dynamic allocation.  One of these per orch.allocate_domain call.
// Tracks the local IPC buffer (aclrtMalloc'd here, freed in
// comm_release_domain_windows) and the device CommContext we materialise for
// the subset.  IPC import refs and EnablePeerAccess routes for this
// allocation are NOT explicitly released — same contract as
// alloc_windows_via_ipc (aclrtResetDevice at finalize reclaims them).
struct DomainAllocation {
    int rank = 0;    // this rank's index within the subset (domain_rank)
    int nranks = 0;  // subset size
    void *local_buf = nullptr;
    CommContext *device_ctx = nullptr;  // aclrtMalloc'd CommContext mirror
    std::unique_ptr<A5UrmaWorkspaceManager> urma_workspace;
};

struct CommHandle_ {
    int rank;
    int nranks;
    std::string rootinfo_path;
    uint64_t run_token = 0;

    // Caller-owned: supplied to comm_init, never created or destroyed here.
    aclrtStream stream = nullptr;
    HcclComm hccl_comm = nullptr;

    CommContext host_ctx{};
    CommContext *device_ctx = nullptr;
    bool owns_device_ctx = false;
    std::vector<CommContext *> derived_contexts;
    std::unordered_map<uint64_t, std::unique_ptr<DomainAllocation>> domain_allocations;
    std::unique_ptr<A5UrmaWorkspaceManager> urma_workspace;
};

// ============================================================================
// Helpers
// ============================================================================

namespace {

static constexpr uint64_t ROOTINFO_MAGIC = 0x50544f5f4843434cULL;  // "PTO_HCCL"

struct RootInfoFileHeader {
    uint64_t magic = ROOTINFO_MAGIC;
    uint64_t run_token = 0;
    uint32_t payload_size = HCCL_ROOT_INFO_BYTES;
    uint32_t reserved = 0;
};

static std::string handshake_dir(const std::string &rootinfo_path) {
    auto last_slash = rootinfo_path.rfind('/');
    if (last_slash == std::string::npos) return ".";
    return rootinfo_path.substr(0, last_slash);
}

static std::string handshake_prefix(const std::string &rootinfo_path) {
    auto last_slash = rootinfo_path.rfind('/');
    return last_slash == std::string::npos ? rootinfo_path : rootinfo_path.substr(last_slash + 1);
}

static std::string run_token_hex(uint64_t run_token) {
    std::ostringstream oss;
    oss << std::hex << run_token;
    return oss.str();
}

static uint64_t make_run_token(int rank) {
    // steady_clock is monotonic and unaffected by NTP or wall-clock jumps;
    // we only need within-host uniqueness for the handshake file naming.
    auto now = std::chrono::time_point_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now())
                   .time_since_epoch()
                   .count();
    uint64_t token = static_cast<uint64_t>(now);
    token ^= static_cast<uint64_t>(getpid()) << 16;
    token ^= static_cast<uint64_t>(rank & 0xFFFF);
    return token;
}

static std::string
barrier_marker_path(const std::string &rootinfo_path, uint64_t run_token, const std::string &tag, int rank) {
    return handshake_dir(rootinfo_path) + "/barrier_" + handshake_prefix(rootinfo_path) + "_" + tag + "_" +
           run_token_hex(run_token) + "_" + std::to_string(rank) + ".ready";
}

static void cleanup_handshake_files(const std::string &rootinfo_path) {
    std::error_code ec;
    std::filesystem::remove(rootinfo_path, ec);

    const std::string prefix = "barrier_" + handshake_prefix(rootinfo_path) + "_";
    const std::string dir = handshake_dir(rootinfo_path);
    for (const auto &entry : std::filesystem::directory_iterator(dir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file(ec)) continue;
        const std::string name = entry.path().filename().string();
        if (name.rfind(prefix, 0) != 0) continue;
        if (name.size() < 6 || name.substr(name.size() - 6) != ".ready") continue;
        std::filesystem::remove(entry.path(), ec);
        ec.clear();
    }
}

static bool
wait_for_rootinfo(const std::string &path, HcclRootInfo *root_info, uint64_t *run_token, int timeout_sec = 120) {
    constexpr int kLogEverySec = 5;
    for (int i = 0; i < timeout_sec * 10; ++i) {
        std::ifstream f(path, std::ios::binary);
        if (f.good()) {
            RootInfoFileHeader header{};
            f.read(reinterpret_cast<char *>(&header), sizeof(header));
            if (!f.good()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            if (header.magic != ROOTINFO_MAGIC || header.payload_size != HCCL_ROOT_INFO_BYTES) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            f.read(root_info->internal, HCCL_ROOT_INFO_BYTES);
            if (!f.good()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            *run_token = header.run_token;
            return true;
        }
        if (i > 0 && i % (kLogEverySec * 10) == 0) {
            LOG_INFO_V0("[comm] wait_for_rootinfo: still waiting (%ds elapsed) path=%s", i / 10, path.c_str());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return false;
}

static bool file_barrier(
    const std::string &rootinfo_path, int rank, int nranks, const std::string &tag, uint64_t run_token,
    int timeout_sec = 120
) {
    std::string my_marker = barrier_marker_path(rootinfo_path, run_token, tag, rank);
    { std::ofstream(my_marker) << "1"; }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_sec);
    for (int r = 0; r < nranks; ++r) {
        std::string marker = barrier_marker_path(rootinfo_path, run_token, tag, r);
        while (true) {
            std::ifstream f(marker);
            if (f.good()) break;
            if (std::chrono::steady_clock::now() >= deadline) {
                LOG_ERROR(
                    "[comm rank %d] file_barrier('%s') timed out after %ds waiting for rank %d", rank, tag.c_str(),
                    timeout_sec, r
                );
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    return true;
}

}  // namespace

// ============================================================================
// API implementation
// ============================================================================

extern "C" CommHandle comm_init(int rank, int nranks, void *stream, const char *rootinfo_path) try {
    if (stream == nullptr) {
        LOG_ERROR("[comm rank %d] comm_init: caller-supplied stream is null", rank);
        return nullptr;
    }
    if (rootinfo_path == nullptr || *rootinfo_path == '\0') {
        LOG_ERROR("[comm rank %d] comm_init: rootinfo_path is null or empty", rank);
        return nullptr;
    }
    if (nranks <= 0 || rank < 0 || rank >= nranks) {
        LOG_ERROR("[comm rank %d] comm_init: invalid rank/nranks (rank=%d, nranks=%d)", rank, rank, nranks);
        return nullptr;
    }
    if (static_cast<uint32_t>(nranks) > COMM_MAX_RANK_NUM) {
        LOG_ERROR("[comm rank %d] comm_init: nranks=%d exceeds COMM_MAX_RANK_NUM=%u", rank, nranks, COMM_MAX_RANK_NUM);
        return nullptr;
    }

    auto *h = new (std::nothrow) CommHandle_{};
    if (!h) return nullptr;

    h->rank = rank;
    h->nranks = nranks;
    h->rootinfo_path = rootinfo_path;
    h->stream = static_cast<aclrtStream>(stream);

    // NOTE: aclInit / aclrtSetDevice / stream creation are intentionally NOT
    // performed here — the caller (DeviceRunner::ensure_acl_ready + a stream
    // it owns) is responsible for them.  This keeps ACL lifecycle ownership
    // in one place (DeviceRunner) and matches HCCL's API shape, which already
    // takes a caller-supplied stream.

    // RootInfo exchange
    HcclRootInfo rootInfo{};
    if (rank == 0) {
        cleanup_handshake_files(h->rootinfo_path);
        h->run_token = make_run_token(rank);
        HcclResult hret = hccl_get_root_info(&rootInfo);
        if (hret != HCCL_SUCCESS) {
            LOG_ERROR("[comm rank 0] HcclGetRootInfo failed: %d", (int)hret);
            delete h;
            return nullptr;
        }
        RootInfoFileHeader header{};
        header.run_token = h->run_token;
        std::string tmp_path = h->rootinfo_path + ".tmp." + std::to_string(getpid());
        std::ofstream fout(tmp_path, std::ios::binary | std::ios::trunc);
        fout.write(reinterpret_cast<const char *>(&header), sizeof(header));
        fout.write(rootInfo.internal, HCCL_ROOT_INFO_BYTES);
        fout.close();
        if (!fout.good() || std::rename(tmp_path.c_str(), h->rootinfo_path.c_str()) != 0) {
            std::remove(tmp_path.c_str());
            delete h;
            return nullptr;
        }
    } else {
        if (!wait_for_rootinfo(h->rootinfo_path, &rootInfo, &h->run_token)) {
            LOG_ERROR("[comm rank %d] Timeout waiting for rootinfo", rank);
            delete h;
            return nullptr;
        }
    }

    if (!file_barrier(h->rootinfo_path, h->rank, h->nranks, "rootinfo_ready", h->run_token)) {
        delete h;
        return nullptr;
    }

    // Init communicator
    HcclResult hret =
        hccl_comm_init_root_info(static_cast<uint32_t>(nranks), &rootInfo, static_cast<uint32_t>(rank), &h->hccl_comm);
    if (hret != HCCL_SUCCESS) {
        LOG_ERROR("[comm rank %d] HcclCommInitRootInfo failed: %d", rank, (int)hret);
        delete h;
        return nullptr;
    }

    if (!file_barrier(h->rootinfo_path, h->rank, h->nranks, "hccl_comm_ready", h->run_token)) {
        hccl_comm_destroy(h->hccl_comm);
        delete h;
        return nullptr;
    }

    return h;
} catch (const std::exception &e) {
    LOG_ERROR("[comm rank %d] comm_init: exception: %s", rank, e.what());
    return nullptr;
} catch (...) {
    LOG_ERROR("[comm rank %d] comm_init: unknown exception", rank);
    return nullptr;
}

namespace {

// Path D: build the per-rank symmetric pool ourselves via the public ACL
// IPC primitives (aclrtMalloc + aclrtIpcMemGetExportKey + SetImportPid +
// ImportByKey). This mirrors HCCL's own internal cross-rank IPC pattern
// (refs/hcomm adapter_rts.cc::hrtIpc* + p2p_mgmt.cc::EnableP2P) so we
// depend only on stable ACL surface, no HCCL-private struct ABI.
// Spike-validated in hw-native-sys/comm-spike; see project memory.

// Default per-rank symmetric pool size when comm_alloc_windows is called
// with win_size == 0. Picked to match the HCCL_BUFFSIZE default of the
// pre-Path-D backend so existing callers see no behavioural change.
constexpr uint64_t kDefaultIpcWinSize = 200ULL * 1024 * 1024;
constexpr size_t kIpcNameLen = 65;
constexpr uint64_t kIpcAnnounceMagic = 0x49504344334d4549ULL;  // "IPCD3MEI"

struct IpcAnnounceFile {
    uint64_t magic;
    int32_t pid;
    uint32_t rank;
    int32_t device_id;  // ACL logic device id this rank is bound to.
    char name[kIpcNameLen];
    char pad[3];  // keep struct size a multiple of 8
};

// Announce file path shares the `barrier_<prefix>_..._<rank>.ready` shape so
// cleanup_handshake_files picks it up alongside the file_barrier markers.
// Without this convention these files would accumulate across re-runs.
static std::string ipc_announce_path(const std::string &rootinfo, int rank, uint64_t run_token) {
    return handshake_dir(rootinfo) + "/barrier_" + handshake_prefix(rootinfo) + "_ipc_announce_" +
           run_token_hex(run_token) + "_" + std::to_string(rank) + ".ready";
}

static bool ipc_write_announce(
    const std::string &rootinfo, int rank, uint64_t run_token, int32_t pid, int32_t device_id, const char *name
) {
    IpcAnnounceFile a{};
    a.magic = kIpcAnnounceMagic;
    a.pid = pid;
    a.rank = static_cast<uint32_t>(rank);
    a.device_id = device_id;
    memcpy(a.name, name, kIpcNameLen);
    std::string p = ipc_announce_path(rootinfo, rank, run_token);
    std::string tmp = p + ".tmp." + std::to_string(getpid());
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f.write(reinterpret_cast<const char *>(&a), sizeof(a));
        if (!f.good()) {
            std::remove(tmp.c_str());
            return false;
        }
    }
    if (std::rename(tmp.c_str(), p.c_str()) != 0) {
        std::remove(tmp.c_str());
        return false;
    }
    return true;
}

static bool ipc_read_announce(
    const std::string &rootinfo, int peer, uint64_t run_token, IpcAnnounceFile *out, int timeout_sec = 60
) {
    std::string p = ipc_announce_path(rootinfo, peer, run_token);
    for (int i = 0; i < timeout_sec * 10; ++i) {
        std::ifstream f(p, std::ios::binary);
        if (f.good()) {
            IpcAnnounceFile a{};
            f.read(reinterpret_cast<char *>(&a), sizeof(a));
            if (f.good() && a.magic == kIpcAnnounceMagic && a.rank == static_cast<uint32_t>(peer)) {
                *out = a;
                return true;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return false;
}

// Fills h->host_ctx with rankId/rankNum/winSize/windowsIn[] via DIY IPC.
// `win_size` is the per-rank pool byte size requested by the caller
// (kDefaultIpcWinSize when 0).
//
// On failure or normal exit, the device-side resources allocated here
// (localBuf via aclrtMalloc, the IPC export key, peer imports, and any
// P2P routes enabled) are NOT explicitly released. DeviceRunner::finalize
// calls aclrtResetDevice at Worker teardown, which reclaims all of the
// above. simpler's current usage is one comm_init/destroy per Worker
// lifetime, so the absence of explicit cleanup does not accumulate
// across runs. If a future caller starts cycling comm contexts within a
// single Worker, explicit teardown will need to land here.
static int alloc_windows_via_ipc(CommHandle h, uint64_t win_size) {
    const int rank = h->rank;
    const int nranks = h->nranks;
    const std::string &rootinfo = h->rootinfo_path;
    const uint64_t run_token = h->run_token;

    // Discover our own device id. Rank != device in general (e.g. simpler's
    // chip_process spawns rank N on whatever device the resource pool gives
    // it). We need real device ids before any cross-rank ACL setup --
    // EnablePeerAccess takes a peer DEVICE id, not a peer rank.
    int32_t myDevice = -1;
    if (aclrtGetDevice(&myDevice) != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] ipc: aclrtGetDevice failed", rank);
        return -1;
    }

    // Allocate local buffer + export its IPC name.
    void *localBuf = nullptr;
    aclError aret = aclrtMalloc(&localBuf, win_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aret != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] ipc: aclrtMalloc -> %d", rank, static_cast<int>(aret));
        return -1;
    }
    char myName[kIpcNameLen]{};
    aret = aclrtIpcMemGetExportKey(localBuf, win_size, myName, kIpcNameLen, 0);
    if (aret != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] ipc: GetExportKey -> %d", rank, static_cast<int>(aret));
        aclrtFree(localBuf);
        return -1;
    }

    // Announce (pid, device, name) and read every peer's announcement.
    const int32_t myPid = static_cast<int32_t>(getpid());
    if (!ipc_write_announce(rootinfo, rank, run_token, myPid, myDevice, myName)) {
        LOG_ERROR("[comm rank %d] ipc: write_announce failed", rank);
        aclrtFree(localBuf);
        return -1;
    }
    std::vector<IpcAnnounceFile> peers(nranks);
    for (int p = 0; p < nranks; ++p) {
        if (p == rank) {
            peers[p].magic = kIpcAnnounceMagic;
            peers[p].pid = myPid;
            peers[p].rank = static_cast<uint32_t>(rank);
            peers[p].device_id = myDevice;
            memcpy(peers[p].name, myName, kIpcNameLen);
            continue;
        }
        if (!ipc_read_announce(rootinfo, p, run_token, &peers[p])) {
            LOG_ERROR("[comm rank %d] ipc: read_announce(peer=%d) timed out", rank, p);
            aclrtFree(localBuf);
            return -1;
        }
    }

    // Now we know every peer's device id. Enable cross-card P2P, then run a
    // best-effort confirmation poll. aclrtDeviceEnablePeerAccess is the
    // operative call: it resolves the peer's physical id via the HCCL adapter
    // and opens the HCCS route. Its success is what matters.
    for (int p = 0; p < nranks; ++p) {
        if (p == rank) continue;
        aclError r = aclrtDeviceEnablePeerAccess(peers[p].device_id, 0);
        if (r != ACL_SUCCESS) {
            // CANN 9.x has no dedicated "already enabled" code, so a non-success
            // here may be a benign re-enable. The poll below is confirmation only.
            LOG_WARN(
                "[comm rank %d] ipc: EnablePeerAccess(peer_dev=%d) -> %d", rank, peers[p].device_id, static_cast<int>(r)
            );
        }
    }
    for (int p = 0; p < nranks; ++p) {
        if (p == rank) continue;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (true) {
            int32_t status = 0;
            aclError r = aclrtDevicePeerAccessStatus(myDevice, peers[p].device_id, &status);
            if (r != ACL_SUCCESS) {
                LOG_ERROR(
                    "[comm rank %d] ipc: PeerAccessStatus(local_dev=%d peer_dev=%d) -> %d", rank, myDevice,
                    peers[p].device_id, static_cast<int>(r)
                );
                aclrtFree(localBuf);
                return -1;
            }
            if (status == 1) break;
            if (std::chrono::steady_clock::now() >= deadline) {
                LOG_WARN(
                    "[comm rank %d] ipc: P2P status unconfirmed peer=%d peer_dev=%d status=%d "
                    "(proceeding after best-effort enable attempt, see device-remap note)",
                    rank, p, peers[p].device_id, status
                );
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // Barrier so every rank has finished its outbound P2P enable+wait.
    if (!file_barrier(rootinfo, rank, nranks, "ipc_p2p_ready", run_token)) {
        aclrtFree(localBuf);
        return -1;
    }

    // Authorize every peer's pid against MY name (batched).
    std::vector<int32_t> peerPids;
    peerPids.reserve(nranks - 1);
    for (int p = 0; p < nranks; ++p) {
        if (p == rank) continue;
        peerPids.push_back(peers[p].pid);
    }
    aret = aclrtIpcMemSetImportPid(myName, peerPids.data(), peerPids.size());
    if (aret != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] ipc: SetImportPid -> %d", rank, static_cast<int>(aret));
        aclrtFree(localBuf);
        return -1;
    }
    if (!file_barrier(rootinfo, rank, nranks, "ipc_auth_done", run_token)) {
        aclrtFree(localBuf);
        return -1;
    }

    // Import every peer's buffer into our local VA space.
    // windowsOut[] is intentionally left zero by the memset below: no kernel
    // path reads it (verified by grep across simpler + pto-isa). The field
    // is kept in CommContext only to preserve byte-equivalence with pto-isa's
    // parallel HcclDeviceContext declaration; removing it is gated on the
    // F4 private-ization decision (see .docs/28.l3-comm/ext.01.pr-774-review.md).
    // host_ctx was value-initialized at handle construction (CommContext{}),
    // and the idempotency guard in comm_alloc_windows prevents a second
    // entry; no re-zero needed before populating it here.
    h->host_ctx.rankId = static_cast<uint32_t>(rank);
    h->host_ctx.rankNum = static_cast<uint32_t>(nranks);
    h->host_ctx.winSize = win_size;
    h->host_ctx.windowsIn[rank] = reinterpret_cast<uint64_t>(localBuf);

    for (int p = 0; p < nranks; ++p) {
        if (p == rank) continue;
        void *peerVa = nullptr;
        // ENABLE_PEER_ACCESS: the imported buffer lives on a peer device, so the
        // import must request cross-device peer access. Without it the driver's
        // halShmemOpenHandle rejects the cross-card handle (drvRetCode=8 ->
        // rtsIpcMemImportByKey 507899). This is the proper API for peer access.
        aret = aclrtIpcMemImportByKey(&peerVa, peers[p].name, ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS);
        if (aret != ACL_SUCCESS) {
            LOG_ERROR(
                "[comm rank %d] ipc: ImportByKey(peer=%d pid=%d) -> %d", rank, p, peers[p].pid, static_cast<int>(aret)
            );
            aclrtFree(localBuf);
            return -1;
        }
        h->host_ctx.windowsIn[p] = reinterpret_cast<uint64_t>(peerVa);
    }

    return 0;
}

// ============================================================================
// Per-domain dynamic allocation (for orch.allocate_domain).
//
// Same Path-D IPC dance as alloc_windows_via_ipc, but on a fresh per-allocation
// local buffer.  Every barrier filename and announce filename is scoped by
// allocation_id so concurrent allocations from different orch.allocate_domain
// calls do not collide.  Participation is by subset (domain_rank within
// rank_count), so non-members of the subset are not involved.
// ============================================================================

// Announce file path scoped by allocation_id so two concurrent allocations
// from different orch calls do not collide.  Same dir + cleanup-friendly
// prefix as the base-comm IPC announce.
static std::string
domain_announce_path(const std::string &rootinfo, uint64_t allocation_id, uint32_t domain_rank, uint64_t run_token) {
    return handshake_dir(rootinfo) + "/barrier_" + handshake_prefix(rootinfo) + "_alloc_" +
           std::to_string(allocation_id) + "_ipc_announce_" + run_token_hex(run_token) + "_" +
           std::to_string(domain_rank) + ".ready";
}

static bool domain_write_announce(
    const std::string &rootinfo, uint64_t allocation_id, uint32_t domain_rank, uint64_t run_token, int32_t pid,
    int32_t device_id, const char *name
) {
    IpcAnnounceFile a{};
    a.magic = kIpcAnnounceMagic;
    a.pid = pid;
    a.rank = domain_rank;
    a.device_id = device_id;
    memcpy(a.name, name, kIpcNameLen);
    std::string p = domain_announce_path(rootinfo, allocation_id, domain_rank, run_token);
    std::string tmp = p + ".tmp." + std::to_string(getpid());
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f.write(reinterpret_cast<const char *>(&a), sizeof(a));
        if (!f.good()) {
            std::remove(tmp.c_str());
            return false;
        }
    }
    if (std::rename(tmp.c_str(), p.c_str()) != 0) {
        std::remove(tmp.c_str());
        return false;
    }
    return true;
}

static bool domain_read_announce(
    const std::string &rootinfo, uint64_t allocation_id, uint32_t peer_domain_rank, uint64_t run_token,
    IpcAnnounceFile *out, int timeout_sec = 60
) {
    std::string p = domain_announce_path(rootinfo, allocation_id, peer_domain_rank, run_token);
    for (int i = 0; i < timeout_sec * 10; ++i) {
        std::ifstream f(p, std::ios::binary);
        if (f.good()) {
            IpcAnnounceFile a{};
            f.read(reinterpret_cast<char *>(&a), sizeof(a));
            if (f.good() && a.magic == kIpcAnnounceMagic && a.rank == peer_domain_rank) {
                *out = a;
                return true;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return false;
}

// Tag helper for allocation-scoped file barriers.  Tag is fed straight into
// `file_barrier`, which already namespaces the marker filename by
// rootinfo prefix + run_token + rank, so adding allocation_id to `tag` is
// enough to keep concurrent allocations from sharing a marker file.
static std::string domain_barrier_tag(uint64_t allocation_id, const char *phase) {
    return std::string("alloc_") + std::to_string(allocation_id) + "_" + phase;
}

static uint64_t urma_workspace_bytes(uint32_t rank_count) {
    using namespace pto::comm::urma;
    constexpr uint32_t qp_num = 1;
    return sizeof(UrmaInfo) +
           static_cast<uint64_t>(rank_count) *
               (2ULL * sizeof(UrmaWQCtx) * qp_num + 2ULL * sizeof(UrmaCqCtx) * qp_num + sizeof(UrmaMemInfo) * qp_num);
}

static bool rank_ids_are_dense_prefix(const uint32_t *rank_ids, size_t rank_count) {
    if (rank_ids == nullptr) return false;
    for (size_t i = 0; i < rank_count; ++i) {
        if (rank_ids[i] != static_cast<uint32_t>(i)) return false;
    }
    return true;
}

static bool init_urma_workspace(
    CommHandle h, uint32_t rank_id, uint32_t rank_count, void *symmetric_addr, uint64_t symmetric_size,
    std::unique_ptr<A5UrmaWorkspaceManager> &workspace
) {
    if (workspace) return workspace->GetWorkspaceAddr() != nullptr;
    if (h == nullptr || h->hccl_comm == nullptr || symmetric_addr == nullptr || symmetric_size == 0 ||
        rank_id >= rank_count) {
        return false;
    }

    auto manager = std::make_unique<A5UrmaWorkspaceManager>();
    if (!manager->Init(h->hccl_comm, rank_id, rank_count, symmetric_addr, symmetric_size)) {
        LOG_WARN(
            "[comm rank %d] URMA workspace init failed (rank_id=%u rank_count=%u size=%llu); "
            "CommContext::workSpace remains 0",
            h->rank, rank_id, rank_count, static_cast<unsigned long long>(symmetric_size)
        );
        return false;
    }
    workspace = std::move(manager);
    return true;
}

static void ensure_base_urma_workspace(CommHandle h) {
    if (h == nullptr || h->urma_workspace) return;
    void *local_buf = reinterpret_cast<void *>(static_cast<uintptr_t>(h->host_ctx.windowsIn[h->rank]));
    if (!init_urma_workspace(
            h, static_cast<uint32_t>(h->rank), static_cast<uint32_t>(h->nranks), local_buf, h->host_ctx.winSize,
            h->urma_workspace
        )) {
        return;
    }
    h->host_ctx.workSpace = reinterpret_cast<uint64_t>(h->urma_workspace->GetWorkspaceAddr());
    h->host_ctx.workSpaceSize = urma_workspace_bytes(static_cast<uint32_t>(h->nranks));
}

// Performs the per-allocation Path-D dance for one subset rank.  rank_ids
// must list participating BASE-COMM rank ids in domain rank order; this
// rank's domain_rank must match its base rank for the same invariant
// alloc_windows_via_ipc relies on (rank_ids[domain_rank] == h->rank).
//
// Failure paths free the local buffer if it was allocated.  IPC imports are
// NOT explicitly torn down on failure — mirrors alloc_windows_via_ipc; ACL
// reset at finalize cleans them up.
static int domain_alloc_via_ipc(
    CommHandle h, uint64_t allocation_id, const uint32_t *rank_ids, size_t rank_count, uint32_t domain_rank,
    uint64_t win_size, DomainAllocation *out
) {
    const std::string &rootinfo = h->rootinfo_path;
    const uint64_t run_token = h->run_token;
    const int subset_n = static_cast<int>(rank_count);
    const int my_dr = static_cast<int>(domain_rank);

    int32_t myDevice = -1;
    if (aclrtGetDevice(&myDevice) != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] alloc_domain: aclrtGetDevice failed", h->rank);
        return -1;
    }

    void *localBuf = nullptr;
    aclError aret = aclrtMalloc(&localBuf, win_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aret != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] alloc_domain: aclrtMalloc -> %d", h->rank, static_cast<int>(aret));
        return -1;
    }
    char myName[kIpcNameLen]{};
    aret = aclrtIpcMemGetExportKey(localBuf, win_size, myName, kIpcNameLen, 0);
    if (aret != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] alloc_domain: GetExportKey -> %d", h->rank, static_cast<int>(aret));
        aclrtFree(localBuf);
        return -1;
    }

    const int32_t myPid = static_cast<int32_t>(getpid());
    if (!domain_write_announce(rootinfo, allocation_id, domain_rank, run_token, myPid, myDevice, myName)) {
        LOG_ERROR("[comm rank %d] alloc_domain: write_announce failed", h->rank);
        aclrtFree(localBuf);
        return -1;
    }
    std::vector<IpcAnnounceFile> peers(subset_n);
    for (int p = 0; p < subset_n; ++p) {
        if (p == my_dr) {
            peers[p].magic = kIpcAnnounceMagic;
            peers[p].pid = myPid;
            peers[p].rank = domain_rank;
            peers[p].device_id = myDevice;
            memcpy(peers[p].name, myName, kIpcNameLen);
            continue;
        }
        if (!domain_read_announce(rootinfo, allocation_id, static_cast<uint32_t>(p), run_token, &peers[p])) {
            LOG_ERROR("[comm rank %d] alloc_domain: read_announce(peer_dr=%d) timed out", h->rank, p);
            aclrtFree(localBuf);
            return -1;
        }
    }

    // Enable cross-card P2P for every domain peer, then a best-effort
    // confirmation poll. The orch-only allocate_domain model has no base
    // comm_alloc_windows to own the P2P route, so each allocation must
    // (idempotently) ensure it. aclrtDeviceEnablePeerAccess is process-global
    // and per device-pair; once any allocation opens a pair, later ones simply
    // observe it. The enable is the operative call (resolves the peer physical
    // id via the HCCL adapter and opens the HCCS route).
    for (int p = 0; p < subset_n; ++p) {
        if (p == my_dr) continue;
        aclError r = aclrtDeviceEnablePeerAccess(peers[p].device_id, 0);
        if (r != ACL_SUCCESS) {
            LOG_WARN(
                "[comm rank %d] alloc_domain: EnablePeerAccess(peer_dev=%d) -> %d", h->rank, peers[p].device_id,
                static_cast<int>(r)
            );
        }
    }
    for (int p = 0; p < subset_n; ++p) {
        if (p == my_dr) continue;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (true) {
            int32_t status = 0;
            aclError r = aclrtDevicePeerAccessStatus(myDevice, peers[p].device_id, &status);
            if (r != ACL_SUCCESS) {
                LOG_ERROR(
                    "[comm rank %d] alloc_domain: PeerAccessStatus(local_dev=%d peer_dev=%d) -> %d", h->rank, myDevice,
                    peers[p].device_id, static_cast<int>(r)
                );
                aclrtFree(localBuf);
                return -1;
            }
            if (status == 1) break;
            if (std::chrono::steady_clock::now() >= deadline) {
                LOG_WARN(
                    "[comm rank %d] alloc_domain: P2P status unconfirmed peer_dr=%d peer_dev=%d status=%d "
                    "(proceeding after best-effort enable attempt, see device-remap note)",
                    h->rank, p, peers[p].device_id, status
                );
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    if (!file_barrier(rootinfo, my_dr, subset_n, domain_barrier_tag(allocation_id, "p2p_ready"), run_token)) {
        aclrtFree(localBuf);
        return -1;
    }

    std::vector<int32_t> peerPids;
    peerPids.reserve(subset_n - 1);
    for (int p = 0; p < subset_n; ++p) {
        if (p == my_dr) continue;
        peerPids.push_back(peers[p].pid);
    }
    aret = aclrtIpcMemSetImportPid(myName, peerPids.data(), peerPids.size());
    if (aret != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] alloc_domain: SetImportPid -> %d", h->rank, static_cast<int>(aret));
        aclrtFree(localBuf);
        return -1;
    }
    if (!file_barrier(rootinfo, my_dr, subset_n, domain_barrier_tag(allocation_id, "auth_done"), run_token)) {
        aclrtFree(localBuf);
        return -1;
    }

    out->rank = my_dr;
    out->nranks = subset_n;
    out->local_buf = localBuf;
    if (rank_ids_are_dense_prefix(rank_ids, rank_count)) {
        (void)init_urma_workspace(
            h, domain_rank, static_cast<uint32_t>(rank_count), localBuf, win_size, out->urma_workspace
        );
    } else {
        LOG_WARN(
            "[comm rank %d] alloc_domain: URMA workspace disabled for non-dense rank mapping "
            "(first supported version requires rank_ids[i] == i)",
            h->rank
        );
    }
    if (!file_barrier(rootinfo, my_dr, subset_n, domain_barrier_tag(allocation_id, "urma_ready"), run_token)) {
        aclrtFree(localBuf);
        return -1;
    }

    CommContext ctx{};
    ctx.rankId = domain_rank;
    ctx.rankNum = static_cast<uint32_t>(subset_n);
    ctx.winSize = win_size;
    if (out->urma_workspace) {
        ctx.workSpace = reinterpret_cast<uint64_t>(out->urma_workspace->GetWorkspaceAddr());
        ctx.workSpaceSize = urma_workspace_bytes(static_cast<uint32_t>(rank_count));
    }
    ctx.windowsIn[my_dr] = reinterpret_cast<uint64_t>(localBuf);
    for (int p = 0; p < subset_n; ++p) {
        if (p == my_dr) continue;
        void *peerVa = nullptr;
        // ENABLE_PEER_ACCESS: the imported buffer lives on a peer device, so the
        // import must request cross-device peer access. Without it the driver's
        // halShmemOpenHandle rejects the cross-card handle (drvRetCode=8 ->
        // rtsIpcMemImportByKey 507899). This is the proper API for peer access.
        aret = aclrtIpcMemImportByKey(&peerVa, peers[p].name, ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS);
        if (aret != ACL_SUCCESS) {
            LOG_ERROR(
                "[comm rank %d] alloc_domain: ImportByKey(peer_dr=%d pid=%d) -> %d", h->rank, p, peers[p].pid,
                static_cast<int>(aret)
            );
            aclrtFree(localBuf);
            return -1;
        }
        ctx.windowsIn[p] = reinterpret_cast<uint64_t>(peerVa);
    }

    void *newDevMem = nullptr;
    aret = aclrtMalloc(&newDevMem, sizeof(CommContext), ACL_MEM_MALLOC_HUGE_FIRST);
    if (aret != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] alloc_domain: ctx aclrtMalloc -> %d", h->rank, static_cast<int>(aret));
        aclrtFree(localBuf);
        return -1;
    }
    aret = aclrtMemcpy(newDevMem, sizeof(CommContext), &ctx, sizeof(CommContext), ACL_MEMCPY_HOST_TO_DEVICE);
    if (aret != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] alloc_domain: ctx Memcpy H2D -> %d", h->rank, static_cast<int>(aret));
        aclrtFree(newDevMem);
        aclrtFree(localBuf);
        return -1;
    }
    out->device_ctx = reinterpret_cast<CommContext *>(newDevMem);
    return 0;
}

}  // namespace

extern "C" int comm_alloc_windows(CommHandle h, size_t win_size, uint64_t *device_ctx_out) try {
    if (!h || !device_ctx_out) return -1;

    // Idempotency guard: comm_alloc_windows is not re-entrant. The localBuf
    // allocated by alloc_windows_via_ipc is owned by the handle's windowsIn[]
    // entries and is only reclaimed at aclrtResetDevice; calling this twice
    // would leak a full per-rank pool. device_ctx is set on first success.
    if (h->device_ctx != nullptr) {
        LOG_ERROR("[comm rank %d] comm_alloc_windows: already allocated on this handle", h->rank);
        return -1;
    }

    // Path D: DIY symmetric pool on stable ACL IPC + EnablePeerAccess.
    // Replaced the prior HcclAllocComResourceByTiling reverse-parse path
    // (broken on CANN 9.0 due to HcclOpResParam ABI drift; see project
    // history). One backend, works on 8.5 and 9.0 unchanged.
    const uint64_t effective_win_size = win_size != 0 ? static_cast<uint64_t>(win_size) : kDefaultIpcWinSize;
    if (alloc_windows_via_ipc(h, effective_win_size) != 0) return -1;

    ensure_base_urma_workspace(h);
    if (!file_barrier(h->rootinfo_path, h->rank, h->nranks, "base_urma_ready", h->run_token)) return -1;

    void *newDevMem = nullptr;
    aclError aRet = aclrtMalloc(&newDevMem, sizeof(CommContext), ACL_MEM_MALLOC_HUGE_FIRST);
    if (aRet != ACL_SUCCESS) return -1;
    aRet = aclrtMemcpy(newDevMem, sizeof(CommContext), &h->host_ctx, sizeof(CommContext), ACL_MEMCPY_HOST_TO_DEVICE);
    if (aRet != ACL_SUCCESS) {
        aclrtFree(newDevMem);
        return -1;
    }
    h->device_ctx = reinterpret_cast<CommContext *>(newDevMem);
    h->owns_device_ctx = true;
    *device_ctx_out = reinterpret_cast<uint64_t>(h->device_ctx);
    return 0;
} catch (const std::exception &e) {
    LOG_ERROR("[comm] comm_alloc_windows: exception: %s", e.what());
    return -1;
} catch (...) {
    LOG_ERROR("[comm] comm_alloc_windows: unknown exception");
    return -1;
}

extern "C" int comm_get_local_window_base(CommHandle h, uint64_t *base_out) {
    if (!h || !base_out) return -1;
    *base_out = h->host_ctx.windowsIn[h->rank];
    return 0;
}

extern "C" int comm_get_window_size(CommHandle h, size_t *size_out) {
    if (!h || !size_out) return -1;
    *size_out = static_cast<size_t>(h->host_ctx.winSize);
    return 0;
}

extern "C" int comm_derive_context(
    CommHandle h, const uint32_t *rank_ids, size_t rank_count, uint32_t domain_rank, size_t window_offset,
    size_t window_size, uint64_t *device_ctx_out
) try {
    if (!h || !rank_ids || !device_ctx_out) return -1;
    if (h->host_ctx.rankNum == 0) {
        LOG_ERROR("[comm rank %d] comm_derive_context: base windows are not allocated", h->rank);
        return -1;
    }
    if (rank_count == 0 || rank_count > COMM_MAX_RANK_NUM || domain_rank >= rank_count) {
        LOG_ERROR(
            "[comm rank %d] comm_derive_context: invalid rank_count=%zu domain_rank=%u", h->rank, rank_count,
            domain_rank
        );
        return -1;
    }
    if (window_offset + window_size > static_cast<size_t>(h->host_ctx.winSize)) {
        LOG_ERROR(
            "[comm rank %d] comm_derive_context: window range [%zu, %zu) exceeds base window size %llu", h->rank,
            window_offset, window_offset + window_size, static_cast<unsigned long long>(h->host_ctx.winSize)
        );
        return -1;
    }

    CommContext ctx{};
    if (rank_ids_are_dense_prefix(rank_ids, rank_count)) {
        ctx.workSpace = h->host_ctx.workSpace;
        ctx.workSpaceSize = h->host_ctx.workSpaceSize;
    } else {
        LOG_WARN(
            "[comm rank %d] comm_derive_context: URMA workspace disabled for non-dense rank mapping "
            "(first supported version requires rank_ids[i] == i)",
            h->rank
        );
    }
    ctx.rankId = domain_rank;
    ctx.rankNum = static_cast<uint32_t>(rank_count);
    ctx.winSize = window_size;
    for (size_t i = 0; i < rank_count; ++i) {
        uint32_t base_rank = rank_ids[i];
        if (base_rank >= static_cast<uint32_t>(h->nranks)) {
            LOG_ERROR(
                "[comm rank %d] comm_derive_context: rank_ids[%zu]=%u out of range [0, %d)", h->rank, i, base_rank,
                h->nranks
            );
            return -1;
        }
        ctx.windowsIn[i] = h->host_ctx.windowsIn[base_rank] + window_offset;
        ctx.windowsOut[i] = h->host_ctx.windowsOut[base_rank] + window_offset;
    }

    void *newDevMem = nullptr;
    aclError aRet = aclrtMalloc(&newDevMem, sizeof(CommContext), ACL_MEM_MALLOC_HUGE_FIRST);
    if (aRet != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] comm_derive_context: aclrtMalloc failed: %d", h->rank, static_cast<int>(aRet));
        return -1;
    }
    aRet = aclrtMemcpy(newDevMem, sizeof(CommContext), &ctx, sizeof(CommContext), ACL_MEMCPY_HOST_TO_DEVICE);
    if (aRet != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] comm_derive_context: aclrtMemcpy H2D failed: %d", h->rank, static_cast<int>(aRet));
        aclrtFree(newDevMem);
        return -1;
    }

    auto *derived = reinterpret_cast<CommContext *>(newDevMem);
    h->derived_contexts.push_back(derived);
    *device_ctx_out = reinterpret_cast<uint64_t>(derived);
    return 0;
} catch (const std::exception &e) {
    LOG_ERROR("[comm] comm_derive_context: exception: %s", e.what());
    return -1;
} catch (...) {
    LOG_ERROR("[comm] comm_derive_context: unknown exception");
    return -1;
}

extern "C" int comm_barrier(CommHandle h) {
    if (!h) return -1;
    // HcclBarrier is synchronous — it blocks until all ranks arrive.
    // Do NOT call aclrtSynchronizeStream after it: HcclBarrier internally
    // switches the thread's ACL context, which invalidates the caller-owned
    // stream for context-checked ACL calls (error 507018).
    HcclResult hret = hccl_barrier(h->hccl_comm, h->stream);
    if (hret != HCCL_SUCCESS) {
        LOG_ERROR("[comm rank %d] HcclBarrier failed: %d", h->rank, static_cast<int>(hret));
        return static_cast<int>(hret);
    }
    return 0;
}

extern "C" int comm_alloc_domain_windows(
    CommHandle h, uint64_t allocation_id, const uint32_t *rank_ids, size_t rank_count, uint32_t domain_rank,
    size_t window_size, uint64_t *device_ctx_out, uint64_t *local_window_base_out
) try {
    if (!h || !rank_ids || !device_ctx_out || !local_window_base_out) return -1;
    if (rank_count == 0 || rank_count > COMM_MAX_RANK_NUM || domain_rank >= rank_count || window_size == 0) {
        LOG_ERROR(
            "[comm rank %d] alloc_domain: bad args (rank_count=%zu domain_rank=%u window_size=%zu)", h->rank,
            rank_count, domain_rank, window_size
        );
        return -1;
    }
    if (h->domain_allocations.count(allocation_id) > 0) {
        LOG_ERROR(
            "[comm rank %d] alloc_domain: allocation_id=%llu already live", h->rank,
            static_cast<unsigned long long>(allocation_id)
        );
        return -1;
    }
    if (rank_ids[domain_rank] != static_cast<uint32_t>(h->rank)) {
        LOG_ERROR(
            "[comm rank %d] alloc_domain: rank_ids[%u]=%u does not match base rank", h->rank, domain_rank,
            rank_ids[domain_rank]
        );
        return -1;
    }
    // The base communicator only needs comm_init to have run (rootinfo_path
    // + run_token are set, used to scope barrier filenames).  We do NOT
    // require comm_alloc_windows on the base in the orch-only model — the
    // dynamic alloc path does its own per-allocation aclrtMalloc + IPC dance.
    if (h->rootinfo_path.empty() || h->hccl_comm == nullptr) {
        LOG_ERROR("[comm rank %d] alloc_domain: base communicator not initialised", h->rank);
        return -1;
    }

    auto alloc = std::make_unique<DomainAllocation>();
    int rc = domain_alloc_via_ipc(h, allocation_id, rank_ids, rank_count, domain_rank, window_size, alloc.get());
    if (rc != 0) return rc;

    // Zero the freshly-allocated local pool so kernels do not observe stale
    // aclrtMalloc bytes (parity with the sim backend's memset).
    aclError aret = aclrtMemset(alloc->local_buf, window_size, 0, window_size);
    if (aret != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] alloc_domain: aclrtMemset -> %d", h->rank, static_cast<int>(aret));
        aclrtFree(alloc->device_ctx);
        aclrtFree(alloc->local_buf);
        return -1;
    }

    *device_ctx_out = reinterpret_cast<uint64_t>(alloc->device_ctx);
    *local_window_base_out = reinterpret_cast<uint64_t>(alloc->local_buf);
    h->domain_allocations.emplace(allocation_id, std::move(alloc));
    return 0;
} catch (const std::exception &e) {
    LOG_ERROR("[comm] alloc_domain: exception: %s", e.what());
    return -1;
} catch (...) {
    LOG_ERROR("[comm] alloc_domain: unknown exception");
    return -1;
}

extern "C" int
comm_release_domain_windows(CommHandle h, uint64_t allocation_id, size_t rank_count, uint32_t domain_rank) try {
    if (!h) return -1;
    auto it = h->domain_allocations.find(allocation_id);
    if (it == h->domain_allocations.end()) {
        LOG_ERROR(
            "[comm rank %d] release_domain: allocation_id=%llu not found", h->rank,
            static_cast<unsigned long long>(allocation_id)
        );
        return -1;
    }
    auto &alloc = it->second;
    if (static_cast<size_t>(alloc->nranks) != rank_count || static_cast<uint32_t>(alloc->rank) != domain_rank) {
        LOG_ERROR(
            "[comm rank %d] release_domain: caller (rank_count=%zu, domain_rank=%u) "
            "disagrees with alloc-time (nranks=%d, rank=%d)",
            h->rank, rank_count, domain_rank, alloc->nranks, alloc->rank
        );
        return -1;
    }
    int rc = 0;
    // Best-effort subset barrier so peers don't free local memory under each
    // other.  If a peer crashed mid-allocation, the timeout returns false and
    // we proceed with local teardown anyway — same shape as comm_destroy.
    if (!file_barrier(
            h->rootinfo_path, static_cast<int>(domain_rank), static_cast<int>(rank_count),
            domain_barrier_tag(allocation_id, "release"), h->run_token
        )) {
        LOG_WARN("[comm rank %d] release_domain: barrier timed out; releasing local state anyway", h->rank);
        rc = -1;
    }

    if (alloc->device_ctx) {
        aclError aret = aclrtFree(alloc->device_ctx);
        if (aret != ACL_SUCCESS && rc == 0) rc = -1;
    }
    alloc->urma_workspace.reset();
    if (alloc->local_buf) {
        aclError aret = aclrtFree(alloc->local_buf);
        if (aret != ACL_SUCCESS && rc == 0) rc = -1;
    }
    h->domain_allocations.erase(it);
    return rc;
} catch (const std::exception &e) {
    LOG_ERROR("[comm] release_domain: exception: %s", e.what());
    return -1;
} catch (...) {
    LOG_ERROR("[comm] release_domain: unknown exception");
    return -1;
}

extern "C" int comm_destroy(CommHandle h) try {
    if (!h) return -1;

    // Final barrier is best-effort: if a peer already crashed we still need to
    // release the local resources we own, so timeout just logs and proceeds.
    int rc = 0;
    if (!file_barrier(h->rootinfo_path, h->rank, h->nranks, "destroy", h->run_token)) {
        LOG_WARN("[comm rank %d] comm_destroy: final barrier timed out; releasing local state anyway", h->rank);
        rc = -1;
    }

    if (h->owns_device_ctx && h->device_ctx) {
        aclrtFree(h->device_ctx);
    }
    for (CommContext *ctx : h->derived_contexts) {
        if (ctx != nullptr) {
            aclrtFree(ctx);
        }
    }
    h->derived_contexts.clear();
    // Reclaim any still-live domain allocations as a safety net.  Caller
    // should release them explicitly via comm_release_domain_windows; this
    // path runs only when an exception or shutdown bypassed that.
    for (auto &kv : h->domain_allocations) {
        auto &alloc = kv.second;
        if (alloc->device_ctx) aclrtFree(alloc->device_ctx);
        alloc->urma_workspace.reset();
        if (alloc->local_buf) aclrtFree(alloc->local_buf);
    }
    h->domain_allocations.clear();
    h->urma_workspace.reset();
    if (h->hccl_comm) {
        HcclResult hret = hccl_comm_destroy(h->hccl_comm);
        if (hret != HCCL_SUCCESS) {
            LOG_ERROR("[comm rank %d] HcclCommDestroy failed: %d", h->rank, static_cast<int>(hret));
            if (rc == 0) rc = -1;
        }
    }

    // NOTE: we do NOT destroy h->stream — it is caller-owned.
    // We also do NOT call aclrtResetDevice / aclFinalize here.  Device/ACL
    // lifecycle belongs to DeviceRunner, whose finalize() releases all
    // device memory before resetting the device and running aclFinalize.

    // Only rank 0 sweeps the on-disk handshake markers, and only if the
    // final barrier succeeded.  Deleting them after a timeout would strand
    // any peer that hasn't observed our marker yet, and leak that peer
    // into the next run with no rootinfo to discover.  Letting cleanup
    // ride on the next rank-0 init is the safer recovery path.
    if (h->rank == 0 && rc == 0) {
        cleanup_handshake_files(h->rootinfo_path);
    }

    delete h;
    return rc;
} catch (const std::exception &e) {
    LOG_ERROR("[comm] comm_destroy: exception: %s", e.what());
    if (h) delete h;
    return -1;
} catch (...) {
    LOG_ERROR("[comm] comm_destroy: unknown exception");
    if (h) delete h;
    return -1;
}
