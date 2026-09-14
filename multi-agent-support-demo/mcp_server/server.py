import json
import os

import mcp.server as MCPServer


mcp = MCPServer(
    "Ticket MCP Server"
)


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "tickets.json"
)


def load_tickets() -> list[dict]:

    with open(
        DATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def save_tickets(
    tickets: list[dict]
) -> None:

    with open(
        DATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            tickets,
            file,
            indent=2,
            ensure_ascii=False
        )


@mcp.tool()
def get_ticket(
    ticket_id: int
) -> dict:
    """
    Liefert ein Ticket anhand seiner ID.
    """

    tickets = load_tickets()

    for ticket in tickets:

        if ticket["id"] == ticket_id:
            return ticket

    return {
        "error": "Ticket nicht gefunden"
    }


@mcp.tool()
def search_tickets(
    search_text: str
) -> list[dict]:
    """
    Sucht Tickets anhand von Kategorie,
    Titel und Beschreibung.
    """

    tickets = load_tickets()

    search_words = (
        search_text
        .lower()
        .split()
    )

    matches = []

    for ticket in tickets:

        ticket_text = (
            ticket.get("category", "")
            + " "
            + ticket.get("title", "")
            + " "
            + ticket.get("description", "")
        ).lower()

        if any(
            word in ticket_text
            for word in search_words
        ):
            matches.append(ticket)

    return matches


@mcp.tool()
def create_ticket(
    category: str,
    title: str,
    description: str
) -> dict:
    """
    Erstellt ein neues Support-Ticket.
    """

    tickets = load_tickets()

    new_id = max(
        (
            ticket["id"]
            for ticket in tickets
        ),
        default=0
    ) + 1

    new_ticket = {
        "id": new_id,
        "category": category,
        "title": title,
        "description": description
    }

    tickets.append(
        new_ticket
    )

    save_tickets(
        tickets
    )

    return new_ticket


if __name__ == "__main__":
    mcp.run()
