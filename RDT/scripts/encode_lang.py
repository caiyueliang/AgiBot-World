import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import torch
import yaml
from models.multimodal_encoder.t5_encoder import T5Embedder


GPU = 0
MODEL_PATH = "google/t5-v1_1-xxl"
CONFIG_PATH = "configs/base.yaml"
SAVE_DIR = "rdt-baseline/lang/"

# Note: if your GPU VRAM is less than 24GB,
# it is recommended to enable offloading by specifying an offload directory.
OFFLOAD_DIR = None  # Specify your offload directory here, ensuring the directory exists.


def get_instruction(task_name):
    
    if task_name == "iros_clear_the_countertop_waste":
        lang = "Pick up the yellow functional beverage can on the table with the left arm.;Threw the yellow functional beverage can into the trash can with the left arm.;Pick up the green carbonated beverage can on the table with the right arm.;Threw the green carbonated beverage can into the trash can with the right arm."
    elif task_name == "iros_restock_supermarket_items":
        lang = "Pick up the brown plum juice from the restock box with the right arm.;Place the brown plum juice on the shelf where the brown plum juice is located with the right arm."
    elif task_name == "iros_clear_table_in_the_restaurant":
        lang = "Pick up the bowl on the table near the right arm with the right arm.;Place the bowl on the plate on the table with the right arm."
    elif task_name == "iros_stamp_the_seal":
        lang = "Pick up the stamp from the ink pad on the table with the right arm.;Stamp the document on the table with the stamp in the right arm.;Place the stamp into the ink pad on the table with the right arm."
    elif task_name == "iros_pack_in_the_supermarket":
        lang = "Pick up the grape juice on the table with the right arm.;Put the grape juice into the felt bag on the table with the right arm."
    elif task_name == "iros_heat_the_food_in_the_microwave":
        lang = "Open the door of the microwave oven with the right arm.;Pick up the plate with bread on the table with the right arm.;Put the plate containing bread into the microwave oven with the right arm.;Push the plate that was not placed properly into the microwave oven the right arm.;Close the door of the microwave oven with the left arm.;Press the start button on the right side of the microwave oven with the right arm."
    elif task_name == "iros_open_drawer_and_store_items":
        lang = "Pull the top drawer of the drawer cabinet with the right arm.;Pick up the Rubik's Cube on the drawer cabinet with the right arm.;Place the Rubik's Cube into the drawer with the right arm.;Push the top drawer of the drawer cabinet with the right arm."
    elif task_name == "iros_pack_moving_objects_from_conveyor":
        lang = "Pick up the hand cream from the conveyor belt with the right arm;Place the hand cream held in the right arm into the box on the table"
    elif task_name == "iros_pickup_items_from_the_freezer":
        lang = "Open the freezer door with the right arm;Pick up the caviar from the freezer with the right arm;Place the caviar held in the right arm into the shopping cart;Close the freezer door with both arms"
    elif task_name == "iros_make_a_sandwich":
        lang = "Pick up the bread slice from the toaster on the table with the right arm;Place the picked bread slice into the plate on the table with the right arm;Pick up the ham slice from the box on the table with the left arm;Place the picked ham slice onto the bread slice in the plate on the table with the left arm;Pick up the lettuce slice from the box on the table with the right arm;Place the picked lettuce slice onto the ham slice in the plate on the table with the right arm;Pick up the bread slice from the toaster on the table with the right arm;Place the bread slice onto the lettuce slice in the plate on the table with the right arm"
    else:
        raise ValueError("task does not exist")
    
    return lang


def main(task_list):
    
    for task in task_list:
        
        INSTRUCTION = get_instruction(task)

        with open(CONFIG_PATH, "r") as fp:
            config = yaml.safe_load(fp)

        device = torch.device(f"cuda:{GPU}")
        text_embedder = T5Embedder(
            from_pretrained=MODEL_PATH,
            model_max_length=config["dataset"]["tokenizer_max_length"],
            device=device,
            use_offload_folder=OFFLOAD_DIR,
        )
        tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model

        tokenized_res = tokenizer(INSTRUCTION, return_tensors="pt", padding="longest", truncation=True)
        tokens = tokenized_res["input_ids"].to(device)
        attn_mask = tokenized_res["attention_mask"].to(device)

        with torch.no_grad():
            text_embeds = text_encoder(input_ids=tokens, attention_mask=attn_mask)["last_hidden_state"].detach().cpu()

        attn_mask = attn_mask.cpu().bool()
        pred = text_embeds[0, attn_mask[0]]

        save_path = os.path.join(SAVE_DIR, f"{task}.pt")
        print(pred.shape)
        torch.save(pred, save_path)

        print(f'"{INSTRUCTION}" is encoded by "{MODEL_PATH}" into shape {pred.shape} and saved to "{save_path}"')


if __name__ == "__main__":
    
    task_list = [
        "iros_clear_the_countertop_waste",
        "iros_restock_supermarket_items",
        "iros_clear_table_in_the_restaurant",
        "iros_stamp_the_seal",
        "iros_heat_the_food_in_the_microwave",
        "iros_open_drawer_and_store_items",
        "iros_pack_moving_objects_from_conveyor",
        "iros_pickup_items_from_the_freezer",
        "iros_make_a_sandwich",
    ]

    main(task_list)
