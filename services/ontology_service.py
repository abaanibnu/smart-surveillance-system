import os
from pathlib import Path
import rdflib
from rdflib import Graph, Namespace, RDF, RDFS, OWL

class OntologyEngine:
    """
    Implements DFD Level 1.5 (P2.2) and UML Class Diagram (Figure 5.13):
    - ontology_file: String
    - expandQueryConcepts(term: String) -> List
    Loads surveillance.owl with RDFLib and performs taxonomic query expansion
    via SPARQL / graph reasoning (e.g. Vehicle -> Car, Bus, Bike, Truck).
    """

    def __init__(self, ontology_file: str = None):
        if ontology_file is None:
            from config import BASE_DIR
            self.ontology_file = str(BASE_DIR / "ontology" / "surveillance.owl")
        else:
            self.ontology_file = str(ontology_file)

        self.graph = Graph()
        self.SURV = Namespace("http://www.semanticweb.org/ontologies/surveillance#")
        
        if os.path.exists(self.ontology_file):
            self.graph.parse(self.ontology_file, format="xml")
            self._is_loaded = True
        else:
            self._is_loaded = False
            print(f"Warning: Ontology file not found at {self.ontology_file}")

        # Build in-memory label lookup cache for fast retrieval
        self.label_to_uri = {}
        self.uri_to_labels = {}
        self._build_cache()

    def _build_cache(self):
        """Pre-indexes class labels and URIs for sub-millisecond expansion."""
        if not self._is_loaded:
            return

        for s, p, o in self.graph.triples((None, RDFS.label, None)):
            label_clean = str(o).strip().lower()
            uri_str = str(s)
            
            if label_clean not in self.label_to_uri:
                self.label_to_uri[label_clean] = []
            self.label_to_uri[label_clean].append(uri_str)

            if uri_str not in self.uri_to_labels:
                self.uri_to_labels[uri_str] = set()
            self.uri_to_labels[uri_str].add(label_clean)

    def expandQueryConcepts(self, term: str) -> list:
        """
        Expands a single concept term into its ontological subclasses,
        synonyms, and related domain concepts.
        Matches the UML Class Diagram method: expandQueryConcepts(term): List
        """
        clean_term = term.strip().lower()
        if not clean_term or not self._is_loaded:
            return [clean_term] if clean_term else []

        expanded = set([clean_term])

        # Step 1: SPARQL Query to retrieve transitive subclasses
        sparql = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        SELECT DISTINCT ?subLabel WHERE {{
            ?concept rdfs:label ?label .
            FILTER(LCASE(STR(?label)) = "{clean_term}")
            
            {{
                ?sub rdfs:subClassOf+ ?concept .
                ?sub rdfs:label ?subLabel .
            }}
            UNION
            {{
                ?concept rdfs:subClassOf ?parent .
                ?parent rdfs:label ?subLabel .
            }}
            UNION
            {{
                ?concept rdfs:label ?subLabel .
            }}
        }}
        """

        try:
            qres = self.graph.query(sparql)
            for row in qres:
                lbl = str(row[0]).strip().lower()
                if lbl:
                    expanded.add(lbl)
        except Exception as e:
            print(f"Ontology SPARQL query error: {e}")

        # Fallback / Direct label cache enhancement
        if clean_term in self.label_to_uri:
            for uri in self.label_to_uri[clean_term]:
                for lbl in self.uri_to_labels.get(uri, []):
                    expanded.add(lbl)

        return sorted(list(expanded))

    def expand_full_query(self, query_text: str) -> dict:
        """
        Analyzes a natural-language query sentence, extracts concepts,
        applies ontology expansion, and builds a composite query string
        without noisy category over-dilution.
        """
        if not query_text:
            return {
                "original_query": "",
                "composite_query": "",
                "detected_concepts": {},
                "expanded_terms": []
            }

        text_lower = query_text.lower().strip()
        words = text_lower.split()
        
        detected_concepts = {}
        all_expanded_terms = set()
        priority_attribute_terms = []

        # Domain Attribute mappings (eyewear, accessories, colors, devices)
        attribute_mappings = {
            "glasses": ("eyewear", ["glasses", "eyeglasses", "sunglasses", "spectacles", "eyewear"]),
            "sunglasses": ("eyewear", ["sunglasses", "shades", "glasses", "eyewear"]),
            "eyeglasses": ("eyewear", ["eyeglasses", "spectacles", "glasses", "eyewear"]),
            "spectacles": ("eyewear", ["spectacles", "glasses", "eyeglasses", "eyewear"]),
            "eyewear": ("eyewear", ["eyewear", "glasses", "sunglasses"]),
            "backpack": ("accessory", ["backpack", "rucksack", "bag", "knapsack"]),
            "rucksack": ("accessory", ["backpack", "rucksack", "bag"]),
            "handbag": ("accessory", ["handbag", "purse", "bag", "tote"]),
            "purse": ("accessory", ["handbag", "purse", "bag"]),
            "cell phone": ("electronic_device", ["cell phone", "mobile", "smartphone", "phone"]),
            "phone": ("electronic_device", ["cell phone", "smartphone", "mobile phone", "phone"]),
            "smartphone": ("electronic_device", ["smartphone", "cell phone", "mobile phone"]),
            "laptop": ("electronic_device", ["laptop", "notebook", "computer"]),
            "black": ("color_attribute", ["black", "dark colored"]),
            "white": ("color_attribute", ["white", "light colored"]),
            "red": ("color_attribute", ["red"]),
            "blue": ("color_attribute", ["blue"]),
            "yellow": ("color_attribute", ["yellow"]),
            "green": ("color_attribute", ["green"]),
            "walking": ("action", ["walking", "pedestrian movement"]),
            "running": ("action", ["running", "jogging", "fast motion"]),
            "standing": ("action", ["standing", "stationary person"])
        }

        has_attribute = False
        for attr_key, (concept_name, syn_list) in attribute_mappings.items():
            if attr_key in text_lower:
                has_attribute = True
                detected_concepts[concept_name] = syn_list
                for s in syn_list:
                    if s not in text_lower and s not in priority_attribute_terms:
                        priority_attribute_terms.append(s)

        # Multi-word phrases check
        multi_words = [
            "cell phone", "mobile phone", "restricted zone", "entrance gate", "parking area",
            "security guard", "suspicious person", "electronic device", "wearing glasses"
        ]
        
        checked_tokens = set()
        for phrase in multi_words:
            if phrase in text_lower:
                sub_concepts = self.expandQueryConcepts(phrase)
                detected_concepts[phrase] = sub_concepts
                all_expanded_terms.update(sub_concepts)
                checked_tokens.update(phrase.split())

        # Single word expansion
        for w in words:
            clean_w = w.strip(".,!?:;\"'()[]{}")
            if clean_w in checked_tokens or len(clean_w) <= 2:
                continue

            # Check if clean_w exists in our ontology labels
            if clean_w in self.label_to_uri:
                concepts = self.expandQueryConcepts(clean_w)
                if len(concepts) > 1 or (len(concepts) == 1 and concepts[0] != clean_w):
                    # If query has modifiers/attributes, restrict person expansion to clean human synonyms
                    if clean_w in ("person", "people", "human", "pedestrian"):
                        if has_attribute:
                            detected_concepts[clean_w] = ["person", "individual", "human", "pedestrian"]
                        else:
                            detected_concepts[clean_w] = ["person", "individual", "pedestrian", "human"]
                    elif clean_w in ("vehicle", "transport", "automotive"):
                        detected_concepts[clean_w] = ["vehicle", "car", "bus", "motorcycle", "bicycle", "truck"]
                        all_expanded_terms.update(["car", "bus", "motorcycle"])
                    else:
                        # Filter out noisy root ontology classes like entity, object, surveillanceentity
                        filtered_concepts = [c for c in concepts if c not in ("entity", "object", "surveillanceentity", "surveillancedomainontology")]
                        detected_concepts[clean_w] = filtered_concepts
                        all_expanded_terms.update(filtered_concepts)

        # Build composite query cleanly without semantic flooding
        if has_attribute and priority_attribute_terms:
            composite_query = f"{query_text} " + " ".join(priority_attribute_terms[:4])
            new_terms = priority_attribute_terms
        else:
            new_terms = [t for t in all_expanded_terms if t not in text_lower and t not in ("entity", "object", "surveillanceentity")]
            if new_terms:
                composite_query = f"{query_text} " + " ".join(new_terms[:5])
            else:
                composite_query = query_text

        return {
            "original_query": query_text,
            "composite_query": composite_query.strip(),
            "detected_concepts": detected_concepts,
            "expanded_terms": new_terms
        }
