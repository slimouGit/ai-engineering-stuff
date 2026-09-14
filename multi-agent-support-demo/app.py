# # python
# import asyncio
# import os
# import sys
#
# from mcp.client import Client
# from mcp.client.stdio import stdio_client, StdioServerParameters
#
# from agents.ticket_agent import TicketAgent
# from agents.knowledge_agent import KnowledgeAgent
# from agents.action_agent import ActionAgent
#
#
# BASE_DIR = os.path.dirname(
#     os.path.abspath(__file__)
# )
#
#
# def create_mcp_server_params():
#     server_script = os.path.join(
#         BASE_DIR,
#         "mcp_server",
#         "server.py"
#     )
#
#     return StdioServerParameters(
#         command=sys.executable,
#         args=[
#             server_script
#         ]
#     )
#
#
# async def process_request(
#     user_input: str,
#     mcp_client
# ):
#
#     ticket_agent = TicketAgent()
#
#     print(
#         "\n=============================="
#     )
#     print(
#         "1. TICKET AGENT"
#     )
#     print(
#         "=============================="
#     )
#
#     ticket_info = ticket_agent.analyze(
#         user_input
#     )
#
#     print(
#         "Kategorie:",
#         ticket_info["category"]
#     )
#     print(
#         "Priorität:",
#         ticket_info["priority"]
#     )
#     print(
#         "Zusammenfassung:",
#         ticket_info["summary"]
#     )
#
#     print(
#         "\n=============================="
#     )
#     print(
#         "2. KNOWLEDGE AGENT / RAG"
#     )
#     print(
#         "=============================="
#     )
#
#     knowledge_agent = KnowledgeAgent()
#
#     knowledge = knowledge_agent.search(
#         ticket_info["summary"]
#     )
#
#     print(
#         "Gefundene Quelle:",
#         knowledge["source"]
#     )
#     print(
#         "Ähnlichkeit:",
#         round(
#             knowledge["score"],
#             3
#         )
#     )
#     print(
#         "\nLösung:"
#     )
#     print(
#         knowledge["answer"]
#     )
#
#     print(
#         "\n=============================="
#     )
#     print(
#         "3. ACTION AGENT / MCP"
#     )
#     print(
#         "=============================="
#     )
#
#     action_agent = ActionAgent(
#         mcp_client
#     )
#
#     action_result = (
#         await action_agent.process(
#             ticket_info
#         )
#     )
#
#     if (
#         action_result["action"]
#         == "EXISTING_TICKET"
#     ):
#         print(
#             "Passendes Ticket gefunden."
#         )
#     else:
#         print(
#             "Kein passendes Ticket gefunden."
#         )
#         print(
#             "Neues Ticket wurde über MCP erstellt."
#         )
#
#     print(
#         "Ticket:",
#         action_result["ticket"]
#     )
#
#     print(
#         "\n=============================="
#     )
#     print(
#         "ERGEBNIS"
#     )
#     print(
#         "=============================="
#     )
#
#     print(
#         "\nProblem:"
#     )
#     print(
#         ticket_info["summary"]
#     )
#
#     print(
#         "\nPriorität:"
#     )
#     print(
#         ticket_info["priority"]
#     )
#
#     print(
#         "\nEmpfohlene Lösung:"
#     )
#     print(
#         knowledge["answer"]
#     )
#
#     print(
#         "\nVerwendete Wissensquelle:"
#     )
#     print(
#         knowledge["source"]
#     )
#
#     print(
#         "\nTicket:"
#     )
#     print(
#         action_result["ticket"]
#     )
#
#
# async def main():
#
#     print(
#         "Multi Agent IT Support Demo"
#     )
#     print(
#         "==========================="
#     )
#
#     server_params = (
#         create_mcp_server_params()
#     )
#
#     async with Client(
#         stdio_client(
#             server_params
#         )
#     ) as mcp_client:
#
#         tools = (
#             await mcp_client.list_tools()
#         )
#
#         tool_names = [
#             tool.name
#             for tool in tools.tools
#         ]
#
#         print(
#             "\nMCP verbunden."
#         )
#         print(
#             "Verfügbare Tools:",
#             tool_names
#         )
#
#         while True:
#
#             user_input = (
#                 await asyncio.to_thread(
#                     input,
#                     "\nDu: "
#                 )
#             ).strip()
#
#             if user_input.lower() in [
#                 "exit",
#                 "quit"
#             ]:
#                 break
#
#             if not user_input:
#                 continue
#
#             await process_request(
#                 user_input,
#                 mcp_client
#             )
#
#
# if __name__ == "__main__":
#     asyncio.run(
#         main()
#     )