import os
from contextlib import asynccontextmanager

import asyncpg
import orjson
import structlog
from fastapi import FastAPI

_pg_pool = None


def get_pg_pool() -> asyncpg.pool.Pool:
    return _pg_pool


async def _init_connection(conn) -> None:
    await conn.set_type_codec(
        "json",
        encoder=lambda v: orjson.dumps(v).decode(),
        decoder=orjson.loads,
        schema="pg_catalog",
    )
    await conn.set_type_codec(
        "jsonb",
        encoder=lambda v: orjson.dumps(v).decode(),
        decoder=orjson.loads,
        schema="pg_catalog",
    )
    await conn.set_type_codec(
        "uuid", encoder=lambda v: str(v), decoder=lambda v: v, schema="pg_catalog"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    如果不需要 lifespan，也可以使用 事件處理器 @app.on_event：

    from fastapi import FastAPI

    app = FastAPI()

    @app.on_event("startup")
    async def startup_event():
        print("應用啟動中...")

    @app.on_event("shutdown")
    async def shutdown_event():
        print("應用關閉中...")
    """
    # 啟動邏輯：應用啟動時執行
    # 1. 設置 Structlog 日誌系統
    """
    # 1. 設置 Structlog 日誌系統
    logger = structlog.get_logger()

    # 2. 綁定上下文
    logger = logger.bind(user_id=123, session_id="abc123")

    # 3. 綁定模塊名稱，方便日誌中顯示模塊名稱
    logger = structlog.get_logger(__name__)
    
    """
    structlog.configure(
        processors=[  # 處理器列表
            structlog.stdlib.filter_by_level,  #過濾日誌級別（filter_by_level）
            structlog.stdlib.PositionalArgumentsFormatter(),  # 格式化位置參數（PositionalArgumentsFormatter）
            structlog.processors.StackInfoRenderer(),  # 堆棧信息渲染器（StackInfoRenderer）
            structlog.processors.UnicodeDecoder(),  # Unicode解碼器（UnicodeDecoder）
            structlog.stdlib.render_to_log_kwargs,  # 渲染到日誌關鍵字（render_to_log_kwargs）
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),  # 日誌工廠（LoggerFactory）
        wrapper_class=structlog.stdlib.BoundLogger,  # 包裝類（BoundLogger）
        cache_logger_on_first_use=True,  # 開啟日誌快取（cache_logger_on_first_use=True），減少頻繁初始化日誌記錄器的性能開銷。
    )

    # 2. 初始化 PostgreSQL 連接池
    global _pg_pool

    _pg_pool = await asyncpg.create_pool(
        database=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.environ["POSTGRES_HOST"],
        port=os.environ["POSTGRES_PORT"],
        init=_init_connection,
    )

    yield  # 將控制權交回 FastAPI

    # 關閉邏輯：應用關閉時執行
    await _pg_pool.close()
    _pg_pool = None

    