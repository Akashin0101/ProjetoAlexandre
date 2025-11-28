import os
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.callbacks import get_openai_callback 

os.environ["OPENAI_API_KEY"] = "api aqui" # Coloquei a chave api juntamente com o link da atividade

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

search_tool = DuckDuckGoSearchRun()

@tool
def analyze_sentiment(text: str) -> str:
    prompt = (
        "Analise o seguinte texto e responda APENAS com uma destas palavras: "
        "'Positivo', 'Negativo' ou 'Neutro'.\n\n"
        f"Texto: {text}"
    )
    resp = llm.invoke([HumanMessage(content=prompt)])
    return resp.content.strip()

tools = [search_tool, analyze_sentiment]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="Você é um agente que pesquisa notícias recentes na web: ao receber um tema, deve buscar a notícia mais recente, gerar um resumo conciso e classificar o sentimento como Positivo/Negativo/Neutro."
)

def main():
    tema = input("Digite o tema para buscar notícias: ").strip()
    if not tema:
        print("Tema vazio. Abortando.")
        return

    query = f"Últimas notícias sobre '{tema}'"

    print("\n--- Processando (com monitoramento de tokens)... ---")
    
    # Monitoramento de tokens (Requisito do trabalho)
    with get_openai_callback() as cb:
        result = agent.invoke({"messages": [HumanMessage(content=query)]})
        
        print(f"\n[Monitoramento] Total Tokens: {cb.total_tokens}")
        print(f"[Monitoramento] Custo Estimado (USD): ${cb.total_cost:.4f}")

    print("\n--- Resposta do agente ---")
    for msg in result.get("messages", []):
        if isinstance(msg, AIMessage):
            print(msg.content)

if __name__ == "__main__":
    main()