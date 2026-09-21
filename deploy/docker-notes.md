# Docker 备选方案

适合无 root、必须用镜像的集群。**本方案未经实际验证**（远程算力未定），
使用时以下面的注意事项为准逐条核对。

## ⚠️ 上游 Dockerfile 与当前安装说明不一致

`repo/docker/` 下有多个 Dockerfile，但与 README 的 Installation 章节**已经脱节**：

| 文件 | 基线 | 问题 |
|---|---|---|
| `docker/Dockerfile.ngc.vllm` | `nvcr.io/nvidia/pytorch:24.05-py3`，torch 2.4.0/cu124 | 文件头注释指向 `vllm0.6.3`，而 README 要求 **vllm 0.11.0** |
| `docker/Dockerfile.sglang` | — | SGLang 路线，`setup.py` 钉死 `torch==2.8.0` |
| `docker/Dockerfile.vllm.sglang.megatron` | — | Megatron 路线 |

**⇒ 不要直接 `docker build` 上游 Dockerfile 就以为能得到与 README 一致的环境。**
以 README 的安装步骤（即 `setup_remote.sh`）为版本依据，Dockerfile 只作结构参考。

## 建议做法：从 NGC 基线自建

以 `nvcr.io/nvidia/pytorch:25.xx-py3`（选与 **vllm 0.11.0** 匹配的 CUDA 版本）为基线，
把 `setup_remote.sh` 的步骤搬进 Dockerfile：

```dockerfile
FROM nvcr.io/nvidia/pytorch:<选与 vllm 0.11.0 匹配的 tag>

# NGC 镜像自带 nv-pytorch fork，先卸载避免与 vllm 的 torch 冲突
RUN pip3 uninstall -y pytorch-quantization pytorch-triton torch torch-tensorrt \
    torchvision xgboost transformer_engine flash_attn apex megatron-core

RUN pip3 install vllm==0.11.0
RUN pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir

COPY repo /workspace/repo
RUN cd /workspace/repo && pip install -e .
```

**要点**：
- NGC 镜像自带的 `flash_attn` 必须先卸载，否则与 2.7.4.post1 冲突。
- flash-attn 编译耗时长且吃内存，构建时设 `MAX_JOBS` 限制并行度（上游 Dockerfile 用 4）。
- 环境包（ALFWorld 等）建议**不要**打进同一镜像，理由同上：依赖冲突。用独立镜像或运行时挂载。

## 验证

镜像构建后，把自检脚本挂进去跑：

```bash
docker run --gpus all -v $PWD:/workspace -w /workspace <image> \
    python deploy/check_env.py --repo /workspace/repo
```

结论应为 ✅（除数据类 ⚠️ 外）。
