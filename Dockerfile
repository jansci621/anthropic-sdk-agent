FROM centos:7

# ── 阿里云 YUM 源 ─────────────────────────────────────────────────────────
RUN sed -e 's|^mirrorlist=|#mirrorlist=|g' \
        -e 's|^#baseurl=http://mirror.centos.org|baseurl=https://mirrors.aliyun.com|g' \
        -i.bak /etc/yum.repos.d/CentOS-Base.repo

# ── 系统依赖 ──────────────────────────────────────────────────────────────
RUN yum makecache && yum install -y \
        epel-release \
        python3 python3-pip \
        gcc gcc-c++ make \
        && yum clean all

# ── 阿里云 PyPI 源 ────────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
    && pip3 config set global.trusted-host mirrors.aliyun.com

# ── 工作目录 & 依赖安装 ──────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# ── 复制项目文件 ──────────────────────────────────────────────────────────
COPY . .

# 数据目录（可通过 volume 持久化）
VOLUME ["/app/data", "/app/knowledge_base", "/app/skills"]

# Web UI 默认端口
EXPOSE 8000

# 启动命令：默认 Web 模式；如需 CLI 改为 ["python3", "main.py"]
CMD ["python3", "main.py", "--web", "--host", "0.0.0.0", "--port", "8000"]
