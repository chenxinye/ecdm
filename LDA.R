#@author: chenxinye

library(NLP)
library(tm)
library(topicmodels)
library(ggplot2)
library(magrittr)

setwd("C:/Users/Administrator/Desktop/E-commerce information mining/history")
pdata.freq <- read.csv("term_freq.csv", stringsAsFactors = FALSE)


pos_corpus <- Corpus(VectorSource(pdata.freq$Var1))

return_documenttermmatrix <- function (corpus){
  param = list(wordLengths = c(1, Inf),bounds = list(global = 5, Inf),removeNumbers = TRUE)
  re = DocumentTermMatrix(corpus,control = param);return (re)}

pos.Matrix <- return_documenttermmatrix(pos_corpus)

meancosine_caculate <- function(Documentmatrix){
  mean_similarity <- c();mean_similarity[1] = 1
  for(i in 2:10){
    control <- list(burnin = 500, iter = 3000, keep = 100)
    Gibbs <- LDA(Documentmatrix, k = i, method = "Gibbs", control = control)
    term <- terms(Gibbs, 50) ;word <- as.vector(term)  ;freq <- table(word) ;unique_word <- names(freq)
    mat <- matrix(rep(0, i * length(unique_word)),nrow = i, ncol = length(unique_word))
    colnames(mat) <- unique_word
    for(k in 1:i){
      for(t in 1:50){mat[k, grep(term[t,k], unique_word)] <- mat[k, grep(term[t, k], unique_word)] + 1}}
    p <- combn(c(1:i), 2);l <- ncol(p);top_similarity <- c()
    for(j in 1:l){
      x <- mat[p[, j][1], ];y <- mat[p[, j][2], ]
      top_similarity[j] <- sum(x * y) / sqrt(sum(x^2) * sum(y ^ 2))}
    mean_similarity[i] <- sum(top_similarity) / l;message("top_num ", i)}
  return(mean_similarity)}

pos_cos <- meancosine_caculate(pos.Matrix)

picture_output <- function(pos_cos) {
  cosdf1 <- data.frame(x = 1:length(pos_cos),meancosine = pos_cos,emotion = rep("positive",10))
  p <- ggplot(cosdf1,aes(x= x,y= meancosine,color = factor(emotion))) 
  p <- p + stat_smooth(se = TRUE) + geom_point();print(p)}

return_writedocument <- function(positive.terms){
  write.csv(positive.terms, "term_LDA.csv", row.names = FALSE)
}


picture_output(pos_cos);cont <- list(burnin = 600, iter = 3000, keep = 100)
pos_gibbs <- LDA(pos.Matrix, k = 3, method = "Gibbs", control = cont)

plot_LDA <- function(terms,data = pdata.freq){
  len1 <- length(terms[,1]);len2 <- length(terms[1,])
  vec <- vector(mode = "logical",length = len2);vec[1] = 0.0
  for (i in 1:len2){count <- 0.0
  for (j in 1:len1){
    freq <- data[which(data$Var1 == terms[j,i]),c("Freq")]
    count <- count + freq}
  count %>% print();vec[i] <- count}
  dataf <- data.frame(vec = vec,topic = as.character(1:len2))
  myLabel = as.vector(dataf$topic)
  myLabel = paste("topic:",myLabel, "(", round(dataf$vec / sum(dataf$vec) * 100, 2), "%)", sep = "")
  
  p = ggplot(dataf, aes(x = "", y = vec, fill = factor(topic))) + 
    geom_bar(stat = "identity", width = 1) +    
    coord_polar(theta = "y") + 
    labs(x = "", y = "", title = "") + 
    theme(axis.ticks = element_blank()) + 
    theme(legend.title = element_blank(), legend.position = "top") + 
    scale_fill_discrete(breaks = dataf$topic, labels = myLabel)
  print(p)
  }

positive.terms <- terms(pos_gibbs, 200)
positive.terms %>% plot_LDA()
print(positive.terms)

return_writedocument(positive.terms)