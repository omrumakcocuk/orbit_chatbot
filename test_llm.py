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
    print("Response object:", response)
    async for chunk in response:
        print("Chunk:", chunk)

asyncio.run(test())
