FROM centos:8

# ── 阿里云 vault 源（CentOS 8 EOL，官方 mirrorlist 已失效）─────────────────
RUN sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-Linux-*.repo \
    && sed -i 's|#baseurl=http://mirror.centos.org|baseurl=https://mirrors.aliyun.com/centos-vault|g' \
           /etc/yum.repos.d/CentOS-Linux-*.repo

# ── 系统依赖 ──────────────────────────────────────────────────────────────
RUN dnf install -y python39 python39-pip gcc gcc-c++ make && dnf clean all \
    && alternatives --set python3 /usr/bin/python3.9 \
    && ln -sf /usr/bin/pip3.9 /usr/bin/pip3

# ── 阿里云 PyPI 源 ────────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
    && pip3 config set global.trusted-host mirrors.aliyun.com

# ── 工作目录 & 依赖安装 ──────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# ── 复制项目文件 ──────────────────────────────────────────────────────────
RUN useradd -m -u 1001 agent

COPY --chown=agent:agent . .

USER agent

# 数据目录（可通过 volume 持久化）
VOLUME ["/app/data", "/app/knowledge_base", "/app/skills"]

# Web UI 默认端口
EXPOSE 8000

# 启动命令：默认 Web 模式；如需 CLI 改为 ["python3", "main.py"]
CMD ["python3", "main.py", "--web", "--host", "0.0.0.0", "--port", "8000"]
