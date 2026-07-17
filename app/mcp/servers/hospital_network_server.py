"""MCP stub server: Hospital Network Directory (port 9001)."""

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

app = FastAPI(title="Hospital Network Directory MCP Server")

# In-memory hospital network data
HOSPITALS = {
    "General Hospital": {
        "in_network": True,
        "address": "123 Main St, New York, NY 10001",
        "rating": 4.2,
        "specialties": ["cardiology", "orthopedics", "emergency"],
        "bed_availability": 45,
        "phone": "+1-212-555-0100",
    },
    "City Care Hospital": {
        "in_network": True,
        "address": "456 Oak Ave, Los Angeles, CA 90001",
        "rating": 4.5,
        "specialties": ["surgery", "oncology", "pediatrics"],
        "bed_availability": 120,
        "phone": "+1-213-555-0200",
    },
    "St. Mary's Medical Center": {
        "in_network": True,
        "address": "789 Pine Rd, Chicago, IL 60601",
        "rating": 4.0,
        "specialties": ["cardiology", "neurology", "maternity"],
        "bed_availability": 80,
        "phone": "+1-312-555-0300",
    },
    "Sunrise Hospital": {
        "in_network": False,
        "address": "321 Elm St, Houston, TX 77001",
        "rating": 3.8,
        "specialties": ["general medicine", "pediatrics"],
        "bed_availability": 30,
        "phone": "+1-713-555-0400",
    },
    "Lakeside Medical": {
        "in_network": True,
        "address": "555 Lake Dr, Phoenix, AZ 85001",
        "rating": 4.7,
        "specialties": ["cardiology", "orthopedics", "neurology", "oncology"],
        "bed_availability": 200,
        "phone": "+1-602-555-0500",
    },
}

# Network coverage per policy (simplified)
POLICY_NETWORK = {
    "P123456": {"tier": "premium", "network": ["General Hospital", "City Care Hospital", "Lakeside Medical"]},
    "P789012": {"tier": "platinum", "network": ["General Hospital", "City Care Hospital", "St. Mary's Medical Center", "Lakeside Medical"]},
    "P654321": {"tier": "basic", "network": ["General Hospital"]},
}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "server": "hospital_network"}


@app.get("/tools")
def list_tools() -> Dict[str, Any]:
    return {
        "tools": [
            {
                "name": "check_hospital_network",
                "description": "Check if a hospital is in-network for a given policy and get coverage details.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "hospital_name": {"type": "string", "description": "Name of the hospital"},
                        "policy_number": {"type": "string", "description": "Policy number to check network coverage"},
                    },
                    "required": ["hospital_name", "policy_number"],
                },
            },
            {
                "name": "get_hospital_details",
                "description": "Get hospital details including address, rating, specialties.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "hospital_name": {"type": "string", "description": "Name of the hospital"},
                    },
                    "required": ["hospital_name"],
                },
            },
        ]
    }


class InvokeRequest(BaseModel):
    tool: str
    arguments: Dict[str, Any] = {}


@app.post("/invoke")
def invoke(req: InvokeRequest) -> Dict[str, Any]:
    if req.tool == "check_hospital_network":
        hospital_name = req.arguments.get("hospital_name", "")
        policy_number = req.arguments.get("policy_number", "")

        hospital = HOSPITALS.get(hospital_name)
        if not hospital:
            return {"error": f"Hospital '{hospital_name}' not found", "found": False}

        policy_network = POLICY_NETWORK.get(policy_number, {"tier": "unknown", "network": []})
        is_in_network = hospital_name in policy_network["network"]

        return {
            "hospital_name": hospital_name,
            "in_network": is_in_network,
            "policy_number": policy_number,
            "policy_tier": policy_network["tier"],
            "hospital_details": {
                "address": hospital["address"],
                "rating": hospital["rating"],
                "specialties": hospital["specialties"],
                "bed_availability": hospital["bed_availability"],
                "phone": hospital["phone"],
            },
            "network_status": "in_network" if is_in_network else "out_of_network",
        }

    elif req.tool == "get_hospital_details":
        hospital_name = req.arguments.get("hospital_name", "")
        hospital = HOSPITALS.get(hospital_name)
        if not hospital:
            return {"error": f"Hospital '{hospital_name}' not found", "found": False}
        return {"hospital_name": hospital_name, **hospital, "found": True}

    return {"error": f"Unknown tool: {req.tool}"}


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=9001, log_level="info")


if __name__ == "__main__":
    main()