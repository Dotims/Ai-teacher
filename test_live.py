import asyncio
from ai_service import client, get_live_config, get_live_model

async def test():
    print(f"Testing model: {get_live_model()}")
    try:
        async with client.aio.live.connect(model=get_live_model(), config=get_live_config()) as session:
            print("Connection successful!")
            await session.send_realtime_input(text="Test")
            async for response in session.receive():
                print("Received response!")
                break
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(test())
