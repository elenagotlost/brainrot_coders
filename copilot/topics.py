"""Rule-based topic tagger for M3Kang (English text). Topic is per problem id,
so it propagates to every translation."""
import re
TOPICS = {
 "logic":        r"\b(liars?|truth|knights?|knaves?|lie[sd]?|true|false|statements?|said|says|claims?|told)\b",
 "combinatorics":r"\b(how many ways|ways|arrangements?|arrange|different (?:results|colou?rs|numbers|ways)|combinations?|choose|pairs?|possible (?:routes|paths|orders?))\b",
 "probability":  r"\b(probability|chance|random(?:ly)?|likely)\b",
 "number_theory":r"\b(digits?|divisible|divisors?|primes?|multiples?|remainder|odd|even|integers?|factors?|palindrom\w*|consecutive)\b",
 "arithmetic":   r"(\d\s*[\+\-×x\*:/÷]\s*\d)|\b(sum|product|total|add(?:s|ed)?|subtract\w*|multipl\w*|divide[sd]?|average|times|result|calculate|equals?)\b",
 "algebra":      r"\b(equations?|unknown|value of [a-z]\b|function|variables?|expression|solutions?|x\s*=|satisf(?:y|ies))\b",
 "plane_geometry":r"\b(triangles?|squares?|rectangles?|circles?|radius|diameter|angles?|°|perimeter|area|polygons?|hexagon|pentagon|segments?|quadrilateral|parallelogram|trapezoid|isosceles|equilateral|vertex|vertices|diagonals?|midpoint)\b",
 "solid_geometry":r"\b(cubes?|cuboid|prisms?|pyramids?|dice|die|faces?|edges?|volume|solid|sphere|cylinder|net|unit cubes?)\b",
 "spatial_visual":r"\b(mirror|rotat\w*|fold\w*|turn\w*|view(?:ed)?|seen from|pieces?|puzzle|which (?:figure|picture|image|piece)|shown|looks? like|shape|pattern|reflect\w*)\b",
 "measurement_time":r"\b(clock|hours?|minutes?|seconds?|days?|weeks?|months?|years? old|age|calendar|o'clock|kg|grams?|litres?|liters?|cm|km|meters?|metres?|speed|km/h|price|euros?|cost|money|coins?)\b",
 "sequences_patterns":r"\b(sequence|pattern|next|continue[sd]?|repeat\w*|series|term|step|rule)\b",
 "grids_tables": r"\b(grid|table|row|column|cells?|board|squares? of the|chessboard|magic square)\b",
 "graphs_paths": r"\b(paths?|routes?|roads?|walks?|maze|network|connected|lines? (?:between|connecting)|map|towns?|cities)\b",
}
CAT = {re.compile(v, re.I): k for k, v in TOPICS.items()}
PRIORITY = ["probability","logic","combinatorics","solid_geometry","plane_geometry","graphs_paths",
            "grids_tables","algebra","number_theory","sequences_patterns","measurement_time","spatial_visual","arithmetic"]
def tag(text, text_only=True):
    t = re.sub(r"^\s*\d+\.\s*", "", text)
    hits = {k: len(rx.findall(t)) for rx, k in CAT.items()}
    hits = {k: v for k, v in hits.items() if v}
    if not text_only: hits["spatial_visual"] = hits.get("spatial_visual", 0) + 0.5
    if not hits: return "other", {}
    # primary: highest count, ties broken by priority (rarer/more specific first)
    primary = max(hits, key=lambda k: (hits[k] + (0.6 if k in PRIORITY[:7] else 0), -PRIORITY.index(k)))
    return primary, hits
