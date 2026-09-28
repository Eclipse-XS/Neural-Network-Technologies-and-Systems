"""Interactive CLI: run from any working directory with the Lab 7 interpreter."""
import argparse
import asyncio
from src.config import Settings
from src.knowledge_base import KnowledgeBase
from src.agent import connect, create_agent, create_session, execute


async def main():
    parser=argparse.ArgumentParser(description='Neural Network Technologies Coursework Knowledge Assistant')
    parser.add_argument('--session',help='Resume an existing SQLite session ID')
    parser.add_argument('--debug',action='store_true')
    args=parser.parse_args()
    client=None
    try:
        settings=Settings()
        client,settings=await connect(settings)
        kb=KnowledgeBase(settings)
        await asyncio.to_thread(kb.initialize)
        agent=create_agent(client,settings,kb)
        session=create_session(settings,args.session)
        print('Coursework Knowledge Assistant\nKnowledge sources: Labs 3, 4, 5, 6')
        print(f'Session: {session.session_id}\n/help for commands',flush=True)
        while True:
            try:
                question=input('You > ').strip()
            except (EOFError,KeyboardInterrupt):
                print('\nSession saved.')
                break
            if not question:
                continue
            if question=='/exit':
                break
            if question=='/help':
                print('/help  /sources  /new  /session  /exit');continue
            if question=='/sources':
                for source in kb.sources():
                    print(source['source_id'],source['workspace_filename'])
                continue
            if question=='/new':
                session=create_session(settings)
                print('Session:',session.session_id);continue
            if question=='/session':
                print('Session:',session.session_id);continue
            if question.startswith('/'):
                print('Unknown command. Use /help.');continue
            result=await execute(agent,settings,question,session=session)
            print('Assistant >',result['final_answer'] if result['status']=='SUCCESS' else result['error'],flush=True)
            if args.debug:
                print('Tools:',', '.join(result['tools_used']) or 'none','| seconds:',result['runtime_seconds'],
                    '| citation IDs:',result['citation'])
    except Exception as exc:
        raise SystemExit(f'Startup failed: {type(exc).__name__}: {exc}. Check LM Studio, Docker/Redis, corpus and local embedding cache.') from exc
    finally:
        if client:
            await client.close()


if __name__=='__main__':
    asyncio.run(main())
