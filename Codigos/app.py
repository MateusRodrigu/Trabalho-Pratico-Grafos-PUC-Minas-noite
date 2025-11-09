"""Entry point de conveniência para Streamlit.

Permite executar:
    streamlit run app.py

Sem duplicar código, apenas delega para `interface/app.py`.
"""

import runpy

# Executa o módulo como se fosse o script principal
runpy.run_module("interface.app", run_name="__main__")
