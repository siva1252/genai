                    DATASET
                       ↓
                Target y exists?
                  /          \
                YES           NO
                 ↓             ↓
           SUPERVISED     UNSUPERVISED
                               ↓
                     What is the goal?
                         /          \
                        ↓            ↓
                  Find Groups    Reduce Features
                        ↓            ↓
                   CLUSTERING       PCA
                        ↓
              ┌─────────┼─────────┐
              ↓         ↓         ↓
           K-Means  Hierarchical  DBSCAN
              ↓         ↓         ↓
              └─────────┼─────────┘
                        ↓
              Evaluate & Compare
                        ↓
                Choose suitable model


Need simple centroid-based clustering
        ↓
K-Means

Need hierarchical relationships / dendrogram
        ↓
Hierarchical Clustering

Need irregular-shaped clusters + noise detection
        ↓
DBSCAN


k-means
Centroids
   ↓
Distance to centroid
   ↓
Assign points
   ↓
Recalculate centroid


Hierarchical Clustering:

Clusters
   ↓
Calculate distance between clusters
   ↓
Merge closest clusters
   ↓
Repeat



#heririahel 
Existing dataset
       ↓
Find which data points are similar
       ↓
Build relationships
       ↓
Create hierarchy
       ↓
Find natural groups

Customers
   ↓
Similar customers
   ↓
Group them
   ↓
Understand customer segments




| Requirement                         | Suitable     |
| ----------------------------------- | ------------ |
| Need centroid-based groups          | K-Means      |
| Need hierarchical relationships     | Hierarchical |
| Need dense groups + noise detection | **DBSCAN**   |



                    DATASET
                       ↓
              Understand requirement
                       ↓
                Unsupervised?
                       ↓
                      YES
                       ↓
                  Need groups?
                       ↓
                      YES
                       ↓
              Inspect data structure
                       ↓
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
   Compact groups   Hierarchy?   Dense + noise?
        ↓              ↓              ↓
    K-Means        Hierarchical     DBSCAN


K-Means
→ Centroid
→ Clusters
→ model.predict() ✅

Hierarchical
→ Distance + merging
→ Hierarchy/Dendrogram
→ model.predict() ❌

DBSCAN
→ Density
→ Clusters + Noise
→ model.predict() ❌    


K-Means
→ "Where does this new customer belong?"

Hierarchical
→ "How are these customers/groups related?"

DBSCAN
→ "Where are the dense groups, and which points are unusual?"