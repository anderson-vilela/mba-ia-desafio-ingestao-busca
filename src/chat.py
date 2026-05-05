from search import search_prompt

EXIT_COMMANDS = {"sair", "exit", "quit", "q"}


def main():
    try:
        chain = search_prompt()
    except Exception as e:
        print(f"Erro ao inicializar o chat: {e}")
        return

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("=" * 60)
    print("Chat com seu PDF (busca semântica)")
    print("Digite sua pergunta e pressione Enter.")
    print("Para sair, digite 'sair' (ou Ctrl+C).")
    print("=" * 60)

    while True:
        print("\nFaça sua pergunta:\n")
        try:
            question = input("PERGUNTA: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando chat.")
            return

        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            print("Encerrando chat.")
            return

        try:
            answer = chain.invoke(question)
        except Exception as e:
            print(f"RESPOSTA: erro ao processar a pergunta: {e}")
            continue

        print(f"RESPOSTA: {answer.strip()}")


if __name__ == "__main__":
    main()
