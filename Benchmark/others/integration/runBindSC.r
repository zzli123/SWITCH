
# %%
library(bindSC)
library(Seurat)
library(zellkonverter)

# %%
args <- commandArgs()
data_dir = args[6]
save_dir = args[7]
n_clus = args[8]
peak_data_name = args[9]

# %%
rna <- readRDS(paste0(data_dir,"/rna.rds"))
genescore <- readRDS(paste0(data_dir,"/GeneScore.rds"))
peak <- readH5AD(paste0(data_dir, "/", peak_data_name))

# %%
rna <- NormalizeData(rna, verbose=FALSE)
rna <- FindVariableFeatures(rna, verbose=FALSE, nfeatures=2000)
rna <- ScaleData(rna, verbose=FALSE)
rna <- RunPCA(rna, npcs = 30, verbose = FALSE)
rna <- FindNeighbors(rna, reduction = "pca", dims = 1:30)
rna <- FindClusters(rna)
print(length(unique((rna$seurat_clusters))))

genescore <- NormalizeData(genescore, verbose=FALSE)
genescore <- FindVariableFeatures(genescore, verbose=FALSE, nfeatures=2000)
genescore <- ScaleData(genescore, verbose=FALSE)
genescore <- RunPCA(genescore, npcs = 30, verbose = FALSE)
genescore <- FindNeighbors(genescore, reduction = "pca", dims = 1:30)
genescore <- FindClusters(genescore)
print(length(unique((genescore$seurat_clusters))))
# %%
X <- rna@assays$RNA$counts
Y <- peak@assays@data$X
Z0 <- genescore@assays$RNA$counts

# %%
gene.overlap <- intersect(rownames(X), rownames(Z0))
cell.overlap <- intersect(colnames(Y), colnames(Z0))

X <- as.matrix(X[gene.overlap,])
Z0 <- as.matrix(Z0[gene.overlap, cell.overlap])
Y <- as.matrix(Y[,cell.overlap])

# %%
out <- dimReduce( dt1 =  X, dt2 = Z0,  K = 30)
x <- out$dt1
z0 <- out$dt2
y  <- dimReduce(dt1 = Y, K=30)

# %%
res <- BiCCA( X = t(x) ,
             Y = t(y), 
             Z0 =t(z0), 
             X.clst = as.vector(rna$seurat_clusters),
             Y.clst = as.vector(genescore$seurat_clusters),
             alpha = 0.1, 
             lambda = 0.7,
             K = 15,
             temp.path  = "out",
             num.iteration = 50,
             tolerance = 0.01,
             save = TRUE,
             parameter.optimize = FALSE,
             block.size = 0)

# %%
library(umap)
dim(res$u)
dim(res$r)
embed = rbind(res$u, res$r)

# %%
write.csv(file=paste0(save_dir,"/BindSC_embed.csv"), embed)
