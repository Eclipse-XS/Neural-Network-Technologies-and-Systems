import asyncio
from agents import SQLiteSession


def test_persistence_and_isolation(tmp_path):
    async def check():
        db = tmp_path / 'memory.sqlite'
        a = SQLiteSession('a', db); b = SQLiteSession('b', db)
        await a.add_items([{'role': 'user', 'content': "Call Lab 5 Vision Lab"}])
        assert len(await a.get_items()) == 1
        assert await b.get_items() == []
        a.close()
        a = SQLiteSession('a', db)
        assert (await a.get_items())[0]['content'] == 'Call Lab 5 Vision Lab'
        await b.add_items([{'role': 'user', 'content': 'Independent turn'}])
        assert len(await a.get_items()) == 1
        a.close(); b.close()
    asyncio.run(check())
