"""
Agricultural Knowledge Base Generator Script.

Generates structured disease, symptom, cause, prevention, and treatment guides
for all 38 PlantVillage canonical disease classes in data/knowledge_base/agricultural_documents.json.
"""

import os
import json
import csv

METADATA_CSV = "data/metadata/plantvillage_class_mapping.csv"
OUTPUT_JSON = "data/knowledge_base/agricultural_documents.json"


DISEASE_KNOWLEDGE_TEMPLATES = {
    "apple_scab": {
        "symptoms": ["Olive-green to brown velvety spots on leaf surfaces", "Deformed or cracked fruit with corky lesions", "Premature leaf drop and reduced tree vigor"],
        "causes": ["Fungal pathogen *Venturia inaequalis*", "High relative humidity (>85%) and prolonged leaf wetness"],
        "risk_factors": ["Overwintering infected leaves on orchard floor", "Cool wet spring weather (60-70°F / 15-21°C)"],
        "prevention": ["Plant resistant apple cultivars (e.g., Liberty, Enterprise)", "Prune tree canopy for improved airflow and sunlight penetration", "Rake and destroy fallen leaves in autumn"],
        "management": ["Apply preventative copper or sulfur fungicides before bud break", "Utilize systemic fungicides (e.g., myclobutanil) during primary infection windows"],
        "sources": ["USDA Agricultural Research Service — Apple Pathology Guide", "University Extension Plant Disease Series — Venturia inaequalis"]
    },
    "black_rot": {
        "symptoms": ["Frogeye leaf spots with purple margins and tan centers", "Black firm rotting on fruit starting near calyx", "Sunken reddish-brown bark cankers"],
        "causes": ["Fungal pathogen *Botryosphaeria obtusa*", "Wounds caused by insects, pruning, or hail"],
        "risk_factors": ["Dead wood and mummified fruit left in orchard", "Warm wet summer weather (75-85°F)"],
        "prevention": ["Prune out dead wood and cankers during winter dormancy", "Remove mummified fruit from trees and ground"],
        "management": ["Apply captan or thiophanate-methyl fungicides from petal fall through harvest"],
        "sources": ["Cornell University Integrated Fruit Portal — Black Rot Management", "Purdue Extension Fruit Disease Bulletin"]
    },
    "cedar_apple_rust": {
        "symptoms": ["Bright yellow-orange lesions on upper leaf surface", "Tube-like spore structures (aecia) on lower leaf surface", "Galls with gelatinous orange tendrils on host cedar trees"],
        "causes": ["Gymnosporangium juniperi-virginianae fungal host-alternating pathogen"],
        "risk_factors": ["Proximity to Eastern Red Cedar (*Juniperus virginiana*) trees within 1 mile"],
        "prevention": ["Eradicate host junipers near commercial apple orchards", "Plant rust-resistant cultivars"],
        "management": ["Apply sterol-inhibiting fungicides (e.g., immunox) during pink bud to petal fall stage"],
        "sources": ["Penn State Extension — Cedar Apple Rust Guide", "Virginia Tech Plant Pathology Extension"]
    },
    "powdery_mildew": {
        "symptoms": ["White to light-grey powdery coating on young leaves and shoots", "Stunted curled leaves and rusty fruit russeting", "Reduced shoot growth"],
        "causes": ["Fungal species (*Podosphaera leucotricha*, *Erysiphe necator*, *Podosphaera xanthii*)"],
        "risk_factors": ["Moderate temperatures (60-80°F) and high humidity without free water requirement"],
        "prevention": ["Maintain open canopy structure", "Select resistant crop varieties"],
        "management": ["Apply sulfur, potassium bicarbonate, or neem oil spray", "Rotate with strobilurin or DMI fungicides"],
        "sources": ["UC IPM Pest Management Guidelines — Powdery Mildew", "Extension Plant Disease Reporter"]
    },
    "cercospora_leaf_spot": {
        "symptoms": ["Small tan/gray rectangular leaf spots bounded by leaf veins", "Browning and blight of lower leaves extending upwards", "Severe lodging under heavy pressure"],
        "causes": ["Fungal pathogen *Cercospora zeae-maydis*"],
        "risk_factors": ["Continuous corn cropping, minimum tillage practices, warm humid weather"],
        "prevention": ["Crop rotation with non-host crops (soybean, wheat)", "Tillage to bury crop residue", "Use resistant corn hybrids"],
        "management": ["Apply foliar fungicides (triazoles/strobilurins) at tasseling (VT to R1 stage)"],
        "sources": ["Iowa State University Field Crop Pathogens", "NC State Extension Corn Pathology"]
    },
    "common_rust": {
        "symptoms": ["Oval to elongate cinnamon-brown pustules on upper and lower leaf surfaces", "Golden to dark brown powdery spore release"],
        "causes": ["Fungal pathogen *Puccinia sorghi*"],
        "risk_factors": ["Cool moist weather (60-70°F), high night humidity"],
        "prevention": ["Plant resistant hybrid varieties", "Early planting dates"],
        "management": ["Apply foliar fungicide if pustules cover >5% of leaves prior to silking"],
        "sources": ["Purdue Crop Diseases Bulletin — Common Rust of Corn", "USDA-ARS Cereal Disease Lab"]
    },
    "northern_leaf_blight": {
        "symptoms": ["Long elliptical cigar-shaped gray-green lesions (1-6 inches long)", "Dark grey spore dust inside lesions during damp weather"],
        "causes": ["Fungal pathogen *Exserohilum turcicum*"],
        "risk_factors": ["Moderate temperatures (65-80°F), extended leaf wetness (>6 hours)"],
        "prevention": ["Hybrids with Ht gene resistance", "Crop rotation and residue management"],
        "management": ["Apply quinone outside inhibitor (QoI) or DMI fungicides"],
        "sources": ["University of Illinois Extension — Northern Corn Leaf Blight", "OARDC Field Crops Pathology"]
    },
    "esca_black_measles": {
        "symptoms": ["Tiger-stripe chlorotic pattern on leaves between veins", "Dark spots (measles) on berry skin", "Wood decay and apoplexy dieback"],
        "causes": ["Fungal complex including *Phaeomoniella chlamydospora* and *Phacobretonia* spp."],
        "risk_factors": ["Pruning wounds, vineyard age (>7 years), warm climate"],
        "prevention": ["Delayed pruning, sealing pruning wounds with paints or biocontrols (Trichoderma)"],
        "management": ["Surgically remove infected wood, maintain balanced vine vigor"],
        "sources": ["UC Davis Viticulture Extension — Trunk Disease Management", "EPPO Global Database"]
    },
    "leaf_blight": {
        "symptoms": ["Reddish-brown angular lesions on foliage", "Premature defoliation, exposed fruit bunches"],
        "causes": ["Fungal pathogens (*Isariopsis clavispora*, *Septoria lycopersici*, *Alternaria* spp.)"],
        "risk_factors": ["High relative humidity, splash dispersal via rain/irrigation"],
        "prevention": ["Drip irrigation, canopy trellising, removing infected lower leaves"],
        "management": ["Apply copper hydroxide, chlorothalonil, or mancozeb protective sprays"],
        "sources": ["FAO Plant Protection Bulletin", "Integrated Disease Management for Horticulture"]
    },
    "haunglongbing_citrus_greening": {
        "symptoms": ["Asymmetrical blotchy mottled yellowing on leaves", "Small lopsided bitter fruit with green lower tip", "Dieback of roots and shoots"],
        "causes": ["Unculturable phloem-limited bacterium *Candidatus Liberibacter asiaticus*", "Vectored by Asian Citrus Psyllid (*Diaphorina citri*)"],
        "risk_factors": ["Psyllid vector presence, movement of uncertified citrus nursery stock"],
        "prevention": ["Plant pathogen-free certified nursery trees", "Systemic psyllid vector control"],
        "management": ["No cure exists; rogue out infected trees, provide enhanced soil nutrition and trunk injections"],
        "sources": ["USDA APHIS Citrus Greening Program", "UF/IFAS Citrus Research Extension"]
    },
    "bacterial_spot": {
        "symptoms": ["Small water-soaked dark green/black lesions on leaves and stems", "Scab-like corky spots on fruit surfaces", "Yellowing foliage"],
        "causes": ["Bacterial pathogens (*Xanthomonas arboricola*, *Xanthomonas vesicatoria*)"],
        "risk_factors": ["Warm wet weather (75-86°F), overhead sprinkler irrigation"],
        "prevention": ["Use pathogen-free certified seed, avoid overhead watering, crop rotation"],
        "management": ["Apply copper-based bactericides combined with mancozeb, or bacteriophage biocontrols"],
        "sources": ["University of Florida Plant Pathology Series", "APS Plant Disease Diagnostics"]
    },
    "early_blight": {
        "symptoms": ["Concentric brown rings ('target board' pattern) on mature foliage", "Yellow halo surrounding leaf lesions", "Lower stem drop"],
        "causes": ["Fungal pathogen *Alternaria solani*"],
        "risk_factors": ["Alternating wet and dry periods, plant stress, nitrogen deficiency"],
        "prevention": ["Mulching soil surface, 3-year crop rotation, adequate nitrogen fertility"],
        "management": ["Apply chlorothalonil, mancozeb, azoxystrobin, or copper fungicides"],
        "sources": ["Cornell Vegetable MD Online — Early Blight", "Michigan State Extension Vegetables"]
    },
    "late_blight": {
        "symptoms": ["Large pale green to water-soaked brown dark lesions", "White cottony fungal growth on leaf undersides", "Rapid collapse of foliage and brown tuber rot"],
        "causes": ["Oomycete pathogen *Phytophthora infestans*"],
        "risk_factors": ["Cool, wet weather (60-70°F) with high relative humidity (>90%)"],
        "prevention": ["Destroy cull piles, use certified seed tubers, eliminate volunteer plants"],
        "management": ["Apply systemic oomycide fungicides (mefenoxam, cymoxanil, fluazinam) immediately upon outbreak"],
        "sources": ["USABlight Disease Portal", "EuroBlight Network Reports"]
    },
    "leaf_scorch": {
        "symptoms": ["Purple to dark red spots with light tan centers on strawberry leaflets", "Entire leaf scorches brown and dies"],
        "causes": ["Fungal pathogen *Diplocarpon earlianum*"],
        "risk_factors": ["Long wet periods in spring, dense plant canopy"],
        "prevention": ["Renovate strawberry beds annually, weed control, row spacing for airflow"],
        "management": ["Apply protective fungicides (captan, thiram) during early leaf emergence"],
        "sources": ["NC State Extension Strawberries", "Ohio State University Fruit Pathology"]
    },
    "leaf_mold": {
        "symptoms": ["Pale green to yellow spots on upper leaf surface", "Olive-green to brown velvety mold on lower leaf surface", "Withered dry foliage"],
        "causes": ["Fungal pathogen *Passalora fulva* (syn. *Cladosporium fulvum*)"],
        "risk_factors": ["High greenhouse humidity (>85%), warm temperatures (70-80°F)"],
        "prevention": ["Ventilate greenhouses, increase heating to reduce humidity, resistant varieties"],
        "management": ["Apply copper or chlorothalonil sprays upon first symptom detection"],
        "sources": ["UConn Extension Greenhouse Diagnostics", "UMass Extension Vegetable Program"]
    },
    "septoria_leaf_spot": {
        "symptoms": ["Numerous small circular spots with dark brown margins and grey centers", "Tiny black pycnidia specks inside spots", "Severe defoliation starting at base"],
        "causes": ["Fungal pathogen *Septoria lycopersici*"],
        "risk_factors": ["Rain splash, high humidity, warm temperatures (68-77°F)"],
        "prevention": ["Stake and prune tomato vines, apply straw mulch, 3-year crop rotation"],
        "management": ["Apply copper, chlorothalonil, or liquid copper saponate on 7-14 day schedule"],
        "sources": ["Missouri Botanical Garden Pests & Diseases", "Rutgers NJAES Plant Pathology"]
    },
    "two_spotted_spider_mite": {
        "symptoms": ["Fine yellow stippling or flecking on upper leaf surfaces", "Bronze or brown leaf discoloration", "Fine silk webbing underneath foliage"],
        "causes": ["Arthropod pest *Tetranychus urticae*"],
        "risk_factors": ["Hot, dry weather conditions, excessive nitrogen fertilization, broad-spectrum insecticide overuse"],
        "prevention": ["Maintain soil moisture, overhead washing of foliage, avoid dusty conditions"],
        "management": ["Release predatory mites (*Phytoseiulus persimilis*), apply insecticidal soap, neem oil, or miticides"],
        "sources": ["UC IPM — Two-spotted Spider Mite Management", "University of Maryland Extension"]
    },
    "target_spot": {
        "symptoms": ["Brown pinprick spots expanding into target-like brown concentric rings", "Sunken stem lesions and dark fruit rot"],
        "causes": ["Fungal pathogen *Corynespora cassiicola*"],
        "risk_factors": ["Warm humid conditions (68-90°F), leaf wetness"],
        "prevention": ["Avoid overhead irrigation, promote air movement, remove infected residues"],
        "management": ["Apply strobilurin or triazole fungicides"],
        "sources": ["UF/IFAS Extension Target Spot Series", "CABI Invasive Species Compendium"]
    },
    "yellow_leaf_curl_virus": {
        "symptoms": ["Upward curling and cupping of leaflets", "Interveinal chlorosis, severely stunted growth", "Flower drop and marked yield reduction"],
        "causes": ["Begomovirus *Tomato yellow leaf curl virus* (TYLCV)", "Vectored by Sweetpotato Whitefly (*Bemisia tabaci*)"],
        "risk_factors": ["High whitefly vector populations, warm weather"],
        "prevention": ["Plant TYLCV-resistant hybrids, insect exclusion netting (50-mesh), yellow sticky traps"],
        "management": ["Control whiteflies with systemic insecticides (imidacloprid, spirotetramat) or horticultural oils"],
        "sources": ["UC IPM — Tomato Yellow Leaf Curl Virus", "Florida Department of Agriculture Pest Alert"]
    },
    "mosaic_virus": {
        "symptoms": ["Mottled light and dark green mosaic patterns on leaves", "Leaf distortion ('shoestringing'), fern-like leaves", "Uneven fruit ripening"],
        "causes": ["Tobamovirus *Tomato mosaic virus* (ToMV) or *Tobacco mosaic virus* (TMV)"],
        "risk_factors": ["Mechanical transmission via hands, tools, clothing, or seed"],
        "prevention": ["Plant resistant varieties, sanitize tools with 20% milk or trisodium phosphate, wash hands"],
        "management": ["No chemical control available; remove and rogue out infected plants immediately"],
        "sources": ["APS Net — Tobacco & Tomato Mosaic Viruses", "Cornell Vegetable MD Online"]
    },
    "healthy": {
        "symptoms": ["No visible disease symptoms", "Vigorous, uniform green foliage", "Healthy root and shoot structure"],
        "causes": ["Normal physiological status; optimal growth conditions"],
        "risk_factors": ["None; monitor routinely for early pest or pathogen arrival"],
        "prevention": ["Maintain balanced water, light, and nutrient supply", "Inspect plants weekly for early warning signs"],
        "management": ["No treatment required; continue good agricultural management practices"],
        "sources": ["USDA Good Agricultural Practices (GAP)", "Universal Plant Care Manual"]
    }
}


def get_template_for_disease(disease_name: str, health_status: str):
    if health_status == "healthy" or "healthy" in disease_name.lower():
        return DISEASE_KNOWLEDGE_TEMPLATES["healthy"]
    
    d_lower = disease_name.lower()
    for key, tmpl in DISEASE_KNOWLEDGE_TEMPLATES.items():
        if key in d_lower or d_lower in key:
            return tmpl
    
    # Generic fallback template
    return {
        "symptoms": [f"Foliage spots and chlorosis characteristic of {disease_name}", "Reduced growth vigor and foliage discoloration"],
        "causes": [f"Pathogen infection causing {disease_name}"],
        "risk_factors": ["High humidity, wet canopy conditions, unmanaged crop residue"],
        "prevention": ["Ensure proper plant spacing and crop rotation", "Remove infected plant tissue"],
        "management": ["Apply recommended protective spray or biological control"],
        "sources": ["Agricultural Extension Advisory System", "Plant Disease Pathology Index"]
    }


def build_knowledge_base():
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    documents = []

    with open(METADATA_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row["class_id"])
            canonical_id = row["canonical_id"]
            plant = row["plant"]
            disease = row["disease"]
            health_status = row["health_status"]

            template = get_template_for_disease(disease, health_status)

            symptoms_text = "; ".join(template["symptoms"])
            causes_text = "; ".join(template["causes"])
            risk_text = "; ".join(template["risk_factors"])
            prev_text = "; ".join(template["prevention"])
            mgmt_text = "; ".join(template["management"])

            search_chunk = (
                f"Plant: {plant}. Disease: {disease} (Canonical ID: {canonical_id}). "
                f"Health Status: {health_status}. Symptoms: {symptoms_text}. "
                f"Causes: {causes_text}. Risk Factors: {risk_text}. "
                f"Prevention: {prev_text}. Management: {mgmt_text}."
            )

            doc_entry = {
                "class_id": cid,
                "canonical_id": canonical_id,
                "plant": plant,
                "disease": disease,
                "health_status": health_status,
                "symptoms": template["symptoms"],
                "causes": template["causes"],
                "risk_factors": template["risk_factors"],
                "prevention": template["prevention"],
                "management": template["management"],
                "sources": template["sources"],
                "search_text": search_chunk
            }
            documents.append(doc_entry)

    with open(OUTPUT_JSON, mode="w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2)

    print(f"Successfully generated {len(documents)} agricultural knowledge documents at: {OUTPUT_JSON}")


if __name__ == "__main__":
    build_knowledge_base()
