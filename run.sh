python grounded_sam_demo.py
  --device cuda
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
  --grounded_checkpoint /home/Rhossolas.Lee/.cache/groundSAM/grounding_dino/groundingdino_swint_ogc.pth
  --sam_hq_checkpoint /home/Rhossolas.Lee/.cache/groundSAM/segment_anything/sam_hq_vit_h.pth
  --sam_checkpoint /home/Rhossolas.Lee/.cache/groundSAM/segment_anything/sam_vit_h_4b8939.pth
  --output_dir outputs
  --box_threshold 0.3 --text_threshold 0.25
  --input_image /home/Rhossolas.Lee/code/IDEA/Grounded-Segment-Anything/assets/demo9.jpg
  --text_prompt "dog."
