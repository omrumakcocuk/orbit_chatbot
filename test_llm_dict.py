import asyncio
import ollama

async def test():
    client = ollama.AsyncClient()
    response = await client.chat(
        model="gemma4:e2b",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
        keep_alive=-1,
        options={
            "num_ctx": 2048, 
            "num_thread": 4, 
            "temperature": 0.4,
            "num_predict": 70
        }
    )
    async for chunk in response:
        try:
            # test dict access
            if isinstance(chunk, dict):
                content = chunk.get("message", {}).get("content", "")
            else:
                content = chunk.message.content
            print("Content:", repr(content))
        except Exception as e:
            print("Exception:", e)

asyncio.run(test())
