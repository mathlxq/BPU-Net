# BPU-Net

A Precise Grain Image Segmentation Method Based on Bidirectional Skip Connections and Continuous VI Loss

# Abstract
In the field of materials science, the precise characterization of grain structures is crucial for understanding the relationship between material properties and microstructures. While deep learning-based 2D image segmentation methods have achieved remarkable progress in grain analysis, existing models generally encounter challenges, such as inadequate utilization of 3D feature continuity and limited boundary segmentation accuracy. To tackle these issues, we propose a deep learning framework termed BPU-Net. Built upon the classic U-Net architecture, this model incorporates two key improvements: First, a novel network structure equipped with bidirectional skip connections and a cross-slice propagation module is designed. By introducing dynamic weights and establishing a bidirectional information transfer mechanism, it enables end-to-end fusion of geometric constraints across consecutive slices. Second, a continuous Variation of Information (VI) metric is innovatively proposed. Through differentiable reconstruction, this metric addresses the gradient vanishing problem induced by traditional discrete metrics, thereby significantly enhancing the discriminative accuracy of boundary pixels. Systematic experiments conducted on the IRON benchmark dataset demonstrate that, by jointly optimizing the network architecture and loss function, this method achieves 68.3% mean Average Precision (mAP) and 77.1% mean Intersection over Union (mIoU) in the grain boundary segmentation task. These results represent improvements of 1.4 and 1.7 percentage points, respectively, compared to the current state-of-the-art methods. Supplementary generalization experiments on a random discrete grain dataset further verify the model’s strong adaptability to scenarios without sequential slice constraints. Quantitative and qualitative analyses confirm that BPU-Net exhibits distinct advantages in submicron grain boundary segmentation tasks, offering a new and effective technical approach for the quantitative analysis of material microstructures.

# DataSet and Running

```bash
# inference
python main.py

# train BPU-Net
python train.py

```



You can download dataset from https://github.com/Keep-Passion/pure_iron_grain_data_sets.


