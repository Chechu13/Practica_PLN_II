from src.retrieval.context_retriever import ContextRetriever
from src.retrieval.web_retriever import WebRetriever

class FallbackRetriever:
    """
    Intenta recuperar contexto localmente. Si la puntuación de confianza 
    es menor al umbral, recurre a la búsqueda web.
    """
    def __init__(self, index_path, device, threshold=0.60):
        self.local_retriever = ContextRetriever(index_path, device=device)
        self.web_retriever = WebRetriever()
        self.threshold = threshold

    def retrieve(self, question):
        # 1. Búsqueda local primero
        local_result = self.local_retriever.retrieve(question)
        
        # 2. Evaluar confianza
        if local_result["score"] >= self.threshold:
            return local_result
            
        # 3. Fallback a Internet si la confianza es baja
        print(f"🌐 Confianza local baja ({local_result['score']:.2f}). Buscando en web: {question}")
        web_result = self.web_retriever.retrieve(question)
        
        # Si la web tampoco devuelve nada útil, nos quedamos con el local original
        if not web_result["context"]:
            return local_result
            
        return web_result
    
    @property
    def context_ids(self):
        return self.local_retriever.context_ids