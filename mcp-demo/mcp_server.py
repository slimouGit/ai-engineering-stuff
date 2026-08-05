import json
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

BASE_DIR = Path(__file__).resolve().parent
TICKET_FILE = BASE_DIR / "tickets.json"

mcp = FastMCP("Lokaler Ticket-MCP-Server")


def load_tickets() -> dict[str, dict[str, Any]]:
    with TICKET_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_tickets(tickets: dict[str, dict[str, Any]]) -> None:
    with TICKET_FILE.open("w", encoding="utf-8") as file:
        json.dump(tickets, file, ensure_ascii=False, indent=2)


@mcp.tool()
def get_ticket(ticket_id: int) -> dict[str, Any]:
    """Lädt ein Ticket anhand seiner numerischen ID."""
    tickets = load_tickets()
    ticket = tickets.get(str(ticket_id))
    if ticket is None:
        return {"success": False, "error": f"Ticket {ticket_id} wurde nicht gefunden."}
    return {"success": True, "ticket": ticket}

@mcp.tool()
def delete_ticket(ticket_id: int) -> dict[str, Any]:
    """Löscht ein Ticket anhand seiner numerischen ID."""
    tickets = load_tickets()
    ticket = tickets.get(str(ticket_id))
    if ticket is None:
        return {"success": False, "error": f"Ticket {ticket_id} wurde nicht gefunden."}
    return {"success": True, "ticket": ticket}

@mcp.tool()
def print_ticket(ticket_id: int) -> dict[str, Any]:
    """Drucke alle Informationen zu einem Ticket anhand seiner numerischen ID."""
    tickets = load_tickets()
    ticket = tickets.get(str(ticket_id))
    if ticket is None:
        return {"success": False, "error": f"Ticket {ticket_id} wurde nicht gefunden."}
    return {"success": True, "ticket": ticket}


@mcp.tool()
def search_tickets(search_text: str) -> dict[str, Any]:
    """Sucht in Titel und Beschreibung aller Tickets."""
    query = search_text.strip().lower()
    if not query:
        return {"success": True, "tickets": []}

    results = []
    for ticket in load_tickets().values():
        text = f"{ticket['title']} {ticket['description']}".lower()
        if query in text:
            results.append(ticket)
    return {"success": True, "tickets": results}


@mcp.tool()
def update_ticket_status(ticket_id: int, new_status: str) -> dict[str, Any]:
    """Ändert den Ticketstatus. Erlaubt: OPEN, IN_PROGRESS, RESOLVED, CLOSED."""
    allowed = {"OPEN", "IN_PROGRESS", "RESOLVED", "CLOSED"}
    normalized = new_status.strip().upper()
    if normalized not in allowed:
        return {"success": False, "error": f"Ungültiger Status. Erlaubt: {sorted(allowed)}"}

    tickets = load_tickets()
    ticket = tickets.get(str(ticket_id))
    if ticket is None:
        return {"success": False, "error": f"Ticket {ticket_id} wurde nicht gefunden."}

    old_status = ticket["status"]
    ticket["status"] = normalized
    save_tickets(tickets)
    return {
        "success": True,
        "ticket_id": ticket_id,
        "old_status": old_status,
        "new_status": normalized,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
