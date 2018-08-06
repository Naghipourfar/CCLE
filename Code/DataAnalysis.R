library(ggplot2)
theme_set(theme_bw())

drug_data = data.table::fread("../../Desktop/Summer 2018/Bioinformatics/CCLE/Data/Drugs_data/17-AAG.csv", header = TRUE, sep = ',')
drug_data$"ActArea" = BBmisc::normalize(x = drug_data$"ActArea", range = c(-1, 1))

ggplot(data = drug_data, aes(rownames(drug_data), ActArea)) + 
  geom_bar(stat = "identity", width = 2, fill = "tomato3") + 
  labs(title = "Normalized Activity Area for each Cell line (17-AAG)") + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.6)) + 
  xlab("Cell Lines") + ylab("Normalized Activity Area")
