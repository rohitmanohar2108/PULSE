from rag.main import RAGApplication
import os
import asyncio

async def test_query():
    rag = RAGApplication(use_sentence_transformer=True)
    data_dir = 'data'

    if os.path.exists(data_dir):
        print('Loading and processing documents...')
        result = rag.load_and_process_documents(data_dir, skip_ocr=True)
        
        if result:
            print('✓ Documents processed successfully!\n')
            
            # Ask a test question
            question = 'What is this document about?'
            print(f'Question: {question}')
            answer = await rag.query_document(question)
            print(f'Answer: {answer}')
        else:
            print('✗ Failed to process documents')
    else:
        print('Data directory not found')

asyncio.run(test_query())
