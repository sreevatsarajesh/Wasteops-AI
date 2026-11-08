GUIDANCE = {
    "cardboard": {"action": "Keep dry and flatten where practical.", "notes": "Remove loose non-paper attachments when your local program requires it. Wet or heavily food-soiled cardboard may not be accepted."},
    "food_organics": {"action": "Check whether food scraps are accepted in a local organics or compost collection.", "notes": "Do not put meat, dairy, oils or compostable packaging into a collection stream unless its operator explicitly accepts them."},
    "glass": {"action": "Keep separate and handle carefully; use a designated glass collection if available.", "notes": "Broken glass can be hazardous. Some programs accept only bottles and jars, not window glass, ceramics or mirrors."},
    "metal": {"action": "Separate from general waste and check local metal or can collection rules.", "notes": "Empty containers where safe. Aerosol cans, batteries and pressurized containers may require special handling."},
    "miscellaneous_trash": {"action": "Check the item label or local waste authority guidance before disposal.", "notes": "This broad category is uncertain; do not assume it is recyclable."},
    "paper": {"action": "Keep clean and dry, then use paper recycling if locally accepted.", "notes": "Tissue, thermal receipts, laminated paper and heavily soiled paper are often excluded."},
    "plastic": {"action": "Check the resin type and your local collection rules; empty and rinse if required.", "notes": "A resin code does not guarantee local recyclability. Films, multilayer packs and small items are frequently excluded."},
    "textile_trash": {"action": "Look for a textile reuse or take-back program before disposal.", "notes": "Wet, moldy or contaminated textiles may not be accepted for reuse or recycling."},
    "vegetation": {"action": "Use a local garden-waste or compost collection where available.", "notes": "Invasive plants, diseased material and treated wood may need special handling."},
}

def get_guidance(label: str) -> dict:
    return {**GUIDANCE.get(label, GUIDANCE["miscellaneous_trash"]), "local_rules_required": True,
            "disclaimer": "General guidance only. Acceptance rules vary by municipality and collection provider; verify locally."}
