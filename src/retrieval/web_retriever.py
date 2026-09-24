from ddgs import DDGS

class WebRetriever:
    """Busca el contexto en internet usando DuckDuckGo."""
    
    def retrieve(self, question):
        try:
            with DDGS() as ddgs:
                # Extraemos los 2 primeros resultados web para tener un contexto más rico
                results = list(ddgs.text(question, max_results=2))
                
                if results:
                    # Unimos los fragmentos de texto (snippets)
                    combined_context = " ".join([res['body'] for res in results])
                    return {
                        "context": combined_context,
                        "context_id": results[0]['href'], # Usamos la URL como identificador
                        "score": 1.0
                    }
        except Exception as e:
            print(f"Error en la búsqueda de '{question}': {e}")
            
        # Fallback si no hay conexión o no hay resultados
        return {
            "context": "",
            "context_id": "none",
            "score": 0.0
        }