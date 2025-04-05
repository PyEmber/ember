"""
Knowledge Graph Example: Demonstrating hallucination mitigation using knowledge graphs

This example shows how to create and use a knowledge graph to verify
and enhance LLM outputs using actual LLM calls through Ember.

To run:
    uv run python src/ember/examples/data/knowledge_graph_example.py
"""

from typing import Dict, List, Set
from dataclasses import dataclass
from collections import defaultdict
import os
from ember.api.operators import Operator, Specification, EmberModel
from ember.api.models import ModelService, ModelRegistry, ModelInfo, ProviderInfo

# Create global model registry and service -- change model as desired
MODEL_REGISTRY = ModelRegistry()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

MODEL_INFO = ModelInfo(
    id="anthropic:claude-3-haiku",
    provider=ProviderInfo(name="anthropic", default_api_key=ANTHROPIC_API_KEY)
)

MODEL_REGISTRY.register_model(MODEL_INFO)

MODEL_SERVICE = ModelService(registry=MODEL_REGISTRY)

@dataclass
class Entity:
    """Represents a node in the knowledge graph."""
    id: str
    name: str
    type: str
    attributes: Dict[str, str]

@dataclass
class Relationship:
    """Represents an edge in the knowledge graph."""
    source_id: str
    target_id: str
    relation_type: str
    confidence: float

class KnowledgeGraph:
    """Simple in-memory knowledge graph implementation."""
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self._relationship_index = defaultdict(list)
        
    def add_entity(self, entity: Entity) -> None:
        self.entities[entity.id] = entity
        
    def add_relationship(self, relationship: Relationship) -> None:
        self.relationships.append(relationship)
        self._relationship_index[relationship.source_id].append(relationship)
        
    def get_related_entities(self, entity_id: str, max_depth: int = 2) -> Set[str]:
        """Get all entities connected to the given entity up to max_depth."""
        related = set()
        to_visit = [(entity_id, 0)]
        visited = set()
        
        while to_visit:
            current_id, depth = to_visit.pop(0)
            if current_id in visited or depth > max_depth:
                continue
                
            visited.add(current_id)
            related.add(current_id)
            
            # Add connected entities
            for rel in self._relationship_index[current_id]:
                to_visit.append((rel.target_id, depth + 1))
                
        return related
        
    def verify_statement(self, subject: str, predicate: str, object: str) -> float:
        """Verify a statement against the knowledge graph."""
        for rel in self.relationships:
            if (rel.source_id == subject and 
                rel.relation_type == predicate and 
                rel.target_id == object):
                return rel.confidence
        return 0.0

class EntityExtractionInput(EmberModel):
    """Input for entity extraction."""
    text: str

class EntityExtractionOutput(EmberModel):
    """Output from entity extraction."""
    entities: List[Entity]

class EntityExtractionSpec(Specification):
    """Specification for entity extraction."""
    input_model: type[EntityExtractionInput] = EntityExtractionInput
    structured_output: type[EntityExtractionOutput] = EntityExtractionOutput
    prompt_template: str = """
    Extract entities from the following text. For each entity, provide:
    1. A unique ID
    2. The entity name
    3. The entity type (person, organization, concept, etc.)
    4. Any relevant attributes

    Text: {text}

    Return the entities in this format:
    [
      {{"id": "unique_id", "name": "Entity Name", "type": "entity_type", "attributes": {{"key": "value"}}}},
      ...
    ]
    """

class EntityExtractorOperator(Operator[EntityExtractionInput, EntityExtractionOutput]):
    """Extracts entities from text using an LLM."""
    
    specification = EntityExtractionSpec()
    
    def __init__(self, model_service: ModelService = MODEL_SERVICE):
        self.model_service = model_service
        self.model = self.model_service.get_model("anthropic:claude-3-haiku")
    
    def forward(self, *, inputs: EntityExtractionInput) -> EntityExtractionOutput:
        """Extract entities from text using an LLM."""
        prompt = self.specification.prompt_template.format(text=inputs.text)
        
        response = self.model(prompt=prompt, max_tokens=1000)
        response_text = str(response)
        
        # Parse the response to extract entities
        try:
            import json
            import re
            
            # Extract JSON array from response
            json_match = re.search(r'\[(.*?)\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                entity_dicts = json.loads(json_str)
                
                entities = []
                for e in entity_dicts:
                    entities.append(Entity(
                        id=e.get("id", "unknown"),
                        name=e.get("name", "Unknown"),
                        type=e.get("type", "unknown"),
                        attributes=e.get("attributes", {})
                    ))
                return EntityExtractionOutput(entities=entities)
        except Exception as e:
            print(f"Error parsing entity extraction response: {e}")
            print(f"Response text: {response_text[:200]}...")
        
        # Fallback: If parsing fails, extract entities using simple pattern matching
        entities = []
        
        # Extract DNA-related entities if they appear in the query
        if "dna" in inputs.text.lower():
            entities.append(Entity(
                id="dna_extracted",
                name="DNA",
                type="molecule",
                attributes={"type": "nucleic_acid"}
            ))
            entities.append(Entity(
                id="nucleotide_extracted",
                name="Nucleotide",
                type="component",
                attributes={"part_of": "DNA"}
            ))
            
        # Extract Einstein-related entities
        if "einstein" in inputs.text.lower():
            entities.append(Entity(
                id="einstein_extracted",
                name="Einstein",
                type="scientist",
                attributes={"field": "physics"}
            ))
            
        # Extract photosynthesis-related entities
        if "photosynthesis" in inputs.text.lower():
            entities.append(Entity(
                id="photosynthesis_extracted",
                name="Photosynthesis",
                type="process",
                attributes={"domain": "biology"}
            ))
            entities.append(Entity(
                id="plants_extracted",
                name="Plants",
                type="organism",
                attributes={"process": "photosynthesis"}
            ))
            
        return EntityExtractionOutput(entities=entities)

class RelationMapperOperator:
    """Maps relationships between entities using an LLM."""
    
    def __init__(self, model_service: ModelService = MODEL_SERVICE):
        self.model_service = model_service
        self.model = self.model_service.get_model("anthropic:claude-3-haiku")
    
    def __call__(self, entities: List[Entity], text: str) -> List[Relationship]:
        """Map relationships between entities using an LLM."""
        if not entities or len(entities) < 2:
            return []
            
        # Create a prompt to identify relationships
        entity_descriptions = "\n".join([f"- {e.name} ({e.type})" for e in entities])
        prompt = f"""
        Identify relationships between the following entities based on this text:
        
        Text: {text}
        
        Entities:
        {entity_descriptions}
        
        For each relationship, provide:
        1. Source entity ID
        2. Target entity ID
        3. Relationship type
        4. Confidence score (0.0 to 1.0)
        
        Return the relationships in this format:
        [
          {{"source_id": "entity1_id", "target_id": "entity2_id", "relation_type": "relationship", "confidence": 0.9}},
          ...
        ]
        """
        
        response = self.model(prompt=prompt, max_tokens=1000)
        response_text = str(response)
        
        # Parse the response to extract relationships
        try:
            import json
            import re
            
            # Extract JSON array from response
            json_match = re.search(r'\[(.*?)\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                rel_dicts = json.loads(json_str)
                
                relationships = []
                for r in rel_dicts:
                    relationships.append(Relationship(
                        source_id=r.get("source_id", "unknown"),
                        target_id=r.get("target_id", "unknown"),
                        relation_type=r.get("relation_type", "unknown"),
                        confidence=float(r.get("confidence", 0.5))
                    ))
                return relationships
        except Exception as e:
            print(f"Error parsing relationship mapping response: {e}")
            
        # Fallback: If parsing fails, create relationships based on entity types
        relationships = []
        
        # Create relationships based on entity types
        for entity1 in entities:
            for entity2 in entities:
                if entity1.id != entity2.id:
                    # DNA-related relationships
                    if entity1.type == "molecule" and entity1.name == "DNA" and entity2.type == "component":
                        relationships.append(Relationship(
                            source_id=entity1.id,
                            target_id=entity2.id,
                            relation_type="contains",
                            confidence=0.9
                        ))
                    
                    # Einstein-related relationships
                    if entity1.name == "Einstein" and entity2.type == "field" and entity2.name == "Physics":
                        relationships.append(Relationship(
                            source_id=entity1.id,
                            target_id=entity2.id,
                            relation_type="contributed_to",
                            confidence=0.95
                        ))
                    
                    # Photosynthesis-related relationships
                    if entity1.name == "Photosynthesis" and entity2.name == "Plants":
                        relationships.append(Relationship(
                            source_id=entity1.id,
                            target_id=entity2.id,
                            relation_type="occurs_in",
                            confidence=0.9
                        ))
        
        return relationships

class FactVerifierOperator:
    """Verifies generated text against knowledge graph."""
    
    def __init__(self, knowledge_graph: KnowledgeGraph, model_service: ModelService = MODEL_SERVICE):
        self.knowledge_graph = knowledge_graph
        self.model_service = model_service
        self.model = self.model_service.get_model("anthropic:claude-3-haiku")
        
    def __call__(self, inputs: Dict[str, str]) -> Dict[str, float]:
        """Verify facts in generated text."""
        generated_text = inputs.get("response", "")
        
        # Extract statements using LLM
        statements = self._extract_statements_with_llm(generated_text)
        results = {}
        
        # Verify statements against the knowledge graph
        for statement in statements:
            subject_id = self._find_entity_id(statement["subject"])
            object_id = self._find_entity_id(statement["object"])
            
            if subject_id and object_id:
                confidence = self.knowledge_graph.verify_statement(
                    subject_id, statement["predicate"], object_id
                )
                statement_text = f"{statement['subject']} {statement['predicate']} {statement['object']}"
                results[statement_text] = confidence
            
        return results
    
    def _extract_statements_with_llm(self, text: str) -> List[Dict[str, str]]:
        """Extract statements from text using an LLM."""
        prompt = f"""
        Extract factual statements from the following text. For each statement, identify:
        1. The subject
        2. The predicate (relationship)
        3. The object
        
        Text: {text}
        
        Return the statements in this format:
        [
          {{"subject": "Einstein", "predicate": "discovered", "object": "relativity"}},
          ...
        ]
        """
        
        response = self.model(prompt=prompt, max_tokens=100)
        
        # Convert response to string
        response_text = str(response)
        
        # Parse the response
        try:
            import json
            import re
            
            # Extract JSON array from response
            json_match = re.search(r'\[(.*?)\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                statements = json.loads(json_str)
                return statements
        except Exception as e:
            print(f"Error parsing statement extraction response: {e}")
        
        return []
    
    def _find_entity_id(self, entity_name: str) -> str:
        """Find entity ID by name (case-insensitive)."""
        for entity_id, entity in self.knowledge_graph.entities.items():
            if entity.name.lower() == entity_name.lower():
                return entity_id
        return None

def create_mock_knowledge_graph() -> KnowledgeGraph:
    """Create a mock knowledge graph for demonstration."""
    kg = KnowledgeGraph()
    
    entities = [
        Entity("einstein_1", "Einstein", "scientist", {"field": "physics"}),
        Entity("physics_1", "Physics", "field", {"domain": "science"}),
        Entity("relativity_1", "Relativity", "theory", {"domain": "physics"}),
        Entity("photosynthesis_1", "Photosynthesis", "process", {"domain": "biology"}),
        Entity("plants_1", "Plants", "organism", {"kingdom": "plantae"}),
        Entity("dna_1", "DNA", "molecule", {"type": "nucleic_acid"}),
    ]
    
    for entity in entities:
        kg.add_entity(entity)
    
    relationships = [
        Relationship("einstein_1", "physics_1", "contributed_to", 1.0),
        Relationship("einstein_1", "relativity_1", "discovered", 1.0),
        Relationship("photosynthesis_1", "plants_1", "occurs_in", 1.0),
    ]
    
    for rel in relationships:
        kg.add_relationship(rel)
    
    return kg

def generate_llm_response(query: str, context: List[Entity], model_service: ModelService = MODEL_SERVICE) -> Dict[str, str]:
    """Generate a response using an LLM with context from the knowledge graph."""
    model = model_service.get_model("anthropic:claude-3-haiku")
    
    # Format context from entities
    context_str = ""
    if context:
        context_str = "Context:\n" + "\n".join([
            f"- {entity.name}: {entity.type}" + 
            (f", {', '.join([f'{k}={v}' for k, v in entity.attributes.items()])}" if entity.attributes else "")
            for entity in context
        ])
    
    prompt = f"""
    {context_str}
    
    Question: {query}
    
    Please provide a concise, factual answer based on the context provided.
    """
    
    response = model(prompt=prompt, max_tokens=500)
    return {"response": response}

def process_query_with_graph(query: str, 
                            entity_extractor: EntityExtractorOperator,
                            relation_mapper: RelationMapperOperator,
                            fact_verifier: FactVerifierOperator,
                            model_service: ModelService) -> Dict:
    """Process a query using knowledge graph and LLMs."""
    
    # Extract entities from query
    extracted_entities = entity_extractor(inputs=EntityExtractionInput(text=query)).entities
    
    # Map relationships between entities
    relationships = relation_mapper(extracted_entities, query)
    
    # Generate response with context
    llm_response = generate_llm_response(query, extracted_entities, model_service)
    
    # Verify response against knowledge graph
    verification_results = fact_verifier(llm_response)
    
    return {
        "entities": extracted_entities,
        "relationships": relationships,
        "response": llm_response["response"],
        "verification_results": verification_results
    }

def main():
    """Run the knowledge graph example."""
    print("Ember Knowledge Graph Example")
    print("============================")
    
    # Use the global model service
    model_service = MODEL_SERVICE
    
    # Create components
    knowledge_graph = create_mock_knowledge_graph()
    entity_extractor = EntityExtractorOperator()
    relation_mapper = RelationMapperOperator()
    fact_verifier = FactVerifierOperator(knowledge_graph)
    
    # Example queries to test
    queries = [
        "Tell me about Einstein's contributions to physics.",
        "What is the relationship between photosynthesis and plants?",
        "Describe the structure of DNA.",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n\n{'='*80}")
        print(f"QUERY {i}: {query}")
        print(f"{'='*80}")
        
        result = process_query_with_graph(
            query=query,
            entity_extractor=entity_extractor,
            relation_mapper=relation_mapper,
            fact_verifier=fact_verifier,
            model_service=model_service
        )
        
        # Display results with improved formatting
        print("\n\nKnowledge Graph Analysis")
        print("-------------------------")
        
        print("\n\nExtracted Entities:")
        if result["entities"]:
            for entity in result["entities"]:
                attrs = ", ".join([f"{k}={v}" for k, v in entity.attributes.items()]) if entity.attributes else "none"
                print(f"  • {entity.name} (ID: {entity.id})")
                print(f"    Type: {entity.type}")
                print(f"    Attributes: {attrs}")
        else:
            print("  No entities extracted")
        
        print("\n\nIdentified Relationships:")
        if result.get("relationships"):
            for rel in result["relationships"]:
                source = next((e.name for e in result["entities"] if e.id == rel.source_id), rel.source_id)
                target = next((e.name for e in result["entities"] if e.id == rel.target_id), rel.target_id)
                print(f"  • {source} {rel.relation_type} {target}")
                print(f"    Confidence: {rel.confidence:.2f}")
        else:
            print("  No relationships identified")
        
        print("\n\nLLM Response:")
        response_text = str(result['response'])
        # Clean up the response text to show only the actual content
        if "data=" in response_text and "raw_output=" in response_text:
            response_text = response_text.split("data=")[1].split(" raw_output=")[0].strip('"')
        print(f"  {response_text.replace(chr(10), chr(10)+'  ')}")
        
        print("\n\nFact Verification")
        if result["verification_results"]:
            high_conf = []
            medium_conf = []
            low_conf = []
            
            for statement, confidence in result["verification_results"].items():
                if confidence > 0.8:
                    high_conf.append((statement, confidence))
                elif confidence > 0.4:
                    medium_conf.append((statement, confidence))
                else:
                    low_conf.append((statement, confidence))
            
            if high_conf:
                print("  VERIFIED FACTS:")
                for statement, confidence in high_conf:
                    print(f"  ✓ {statement}")
                    print(f"    Confidence: {confidence:.2f}")
            
            if medium_conf:
                print("\n  UNCERTAIN CLAIMS:")
                for statement, confidence in medium_conf:
                    print(f"  ? {statement}")
                    print(f"    Confidence: {confidence:.2f}")
            
            if low_conf:
                print("\n  POTENTIAL HALLUCINATIONS:")
                for statement, confidence in low_conf:
                    print(f"  ✗ {statement}")
                    print(f"    Confidence: {confidence:.2f}")
        else:
            print("  No verifiable statements found in response")
        
        print("\n\nKnowledge Graph Statistics:")
        print(f"  • Total entities in graph: {len(knowledge_graph.entities)}")
        print(f"  • Total relationships in graph: {len(knowledge_graph.relationships)}")
        print(f"  • Entities in response: {len(result['entities'])}")
        print(f"  • Verified facts: {len([s for s, c in result['verification_results'].items() if c > 0.8])}")
        print(f"  • Potential hallucinations: {len([s for s, c in result['verification_results'].items() if c < 0.4])}")

if __name__ == "__main__":
    main()
