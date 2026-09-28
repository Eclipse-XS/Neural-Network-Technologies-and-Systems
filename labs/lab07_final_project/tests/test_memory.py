import asyncio
from labs.lab07_final_project.src.agent import create_session
from labs.lab07_final_project.src.config import Settings


def test_sqlite_persistence_and_isolation(tmp_path):
    async def check():
        settings=Settings(session_db=tmp_path/'memory.sqlite')
        a=create_session(settings,'memory');b=create_session(settings,'isolation')
        await a.add_items([{'role':'user','content':'What model was used in Lab 6?'}])
        reopened=create_session(settings,'memory')
        assert (await reopened.get_items())[0]['content']=='What model was used in Lab 6?'
        assert await b.get_items()==[]
        await b.add_items([{'role':'user','content':'What was its weakness?'}])
        assert len(await a.get_items())==1
    asyncio.run(check())
