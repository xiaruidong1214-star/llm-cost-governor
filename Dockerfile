# 单阶段镜像：先用构建器装依赖，再把依赖目录拷进最终镜像，避免把编译工具链留在运行镜像里。
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /src
COPY pyproject.toml README.md ./
COPY costgovernor ./costgovernor
# --prefix 把依赖装到独立目录，便于整体拷贝
RUN pip install --prefix=/install .

FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/usr/local/lib/python3.12/site-packages

COPY --from=builder /install /usr/local

# 非 root 用户运行
RUN groupadd --system --gid 10001 costgov \
 && useradd --system --uid 10001 --gid costgov --create-home costgov

WORKDIR /app
COPY --chown=costgov:costgov pyproject.toml README.md ./
COPY --chown=costgov:costgov costgovernor ./costgovernor

USER costgov

# 只做导入自检，不连接任何外部服务
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD ["python", "-c", "import costgovernor; print(costgovernor.__version__)"]

ENTRYPOINT ["python", "-m", "costgovernor.cli"]
CMD ["--help"]
