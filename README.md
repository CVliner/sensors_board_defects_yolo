### Introduction

The purpose of this project is to detect the defects of new circuit boards, before they delivered to customer.

### Dataset
Circuit dataset was used for the Circuit defects detection project. Dataset containing 1500 images with 6 kinds of defects (missing hole, mouse bite, open circuit, short, spur, spurious copper) for the use of detection, classification and registration tasks.

### Preprocessing

The preprocessing steps of the proposed project are following:

- Mouse bite (If the portion of circuit contains some residue)

- Short (If two portions of circuits are connected accidently)

- Missing hole (If the portion of circuit is emerged)

- Open Circuit (If the metal of the circuit is damaged)

- Spur (If the surface of the circuit is damaged)

- Spurious copper (If the portion of the circuit is broken)

- Unclassified (Any defect does not belong to above classes)

### Model Training
For the Circuit defects Detection, Yolo V5 model was trained with the annotated images. The details of the model training are following:

- Use 1500 Images

### Results
- Used 300 Samples for Evaluation

- Calculation of Mean Average Precision

- Got 0.7 Mean Average Precision of all classes


