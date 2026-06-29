import asyncio

async def test():
    try:
        import websockets
        async with websockets.connect('ws://localhost:8000/ws') as ws:
            print('CONNECTED OK')
            await ws.send('{"sender":"test","text":"hello","timestamp":"2026-01-01T00:00:00Z"}')
            print('SENT OK')
    except ImportError:
        print('websockets not installed, testing with urllib')
        import urllib.request
        try:
            urllib.request.urlopen('http://localhost:8000/ws')
        except Exception as e:
            print(f'HTTP to /ws: {e}')
    except Exception as e:
        print(f'ERROR: {type(e).__name__}: {e}')

asyncio.run(test())
