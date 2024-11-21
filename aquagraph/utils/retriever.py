import json
from typing import Dict, List

from langchain_community.retrievers import AzureAISearchRetriever
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document


class CustomAzureAISearchRetriever(AzureAISearchRetriever):
    def _parse_metadata(self, raw_metadata: Dict) -> Dict:
        """Parse nested metadata JSON string into dictionary."""
        if isinstance(raw_metadata.get("metadata"), str):
            try:
                parsed_metadata = json.loads(raw_metadata["metadata"])
                return {
                    "title": parsed_metadata.get("title", ""),
                    "source": parsed_metadata.get("source", ""),
                    "page": parsed_metadata.get("page", ""),
                }
            except json.JSONDecodeError:
                return raw_metadata
        return raw_metadata

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        search_results = self._search(query)

        documents = []
        for result in search_results:
            content = result.pop(self.content_key)
            parsed_metadata = self._parse_metadata(result)
            documents.append(Document(page_content=content, metadata=parsed_metadata))
        return documents

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        search_results = await self._asearch(query)

        documents = []
        for result in search_results:
            content = result.pop(self.content_key)
            parsed_metadata = self._parse_metadata(result)
            documents.append(Document(page_content=content, metadata=parsed_metadata))
        return documents
