if(FALSE){
  # -*- coding: utf-8 -*-
  "
  Created on Tue Nov 26 21:27:12 2019
  @author: chenxinye
  "
}

library(wordcloud2)
library(magrittr)
library(jiebaR)

setwd('I:/E-commerce information mining/org_data')
df_ps <- read.csv("100003142993po.csv", header=F, encoding = 'UTF-8')

cutter <- worker(type = "tag", stop_word = "dict/stoplist.txt")


delete_nonuseword <- function(comment,vector,bool =T) {
  reviews = comment$V1
  if (bool){
    reviews = gsub("[a-zA-Z0-9]", "", reviews)
    }
  for (i in vector){
    reviews = gsub(as.character(i), "", reviews)
    }
  reviews = gsub("[&|;]", "", reviews)
  reviews = gsub("\\\\", "", reviews)
  return (reviews)
}



return_segment <- function(df,vector){
  reviews <- delete_nonuseword(df,vector,T)
  seg_word <- list()
  for(i in 1:length(reviews)){
    seg_word[[i]] <- segment(reviews[i], cutter)
  }
  return(seg_word)
}

return_result <- function(seg_word){
  n_word <- sapply(seg_word, length)
  index <- rep(1:length(seg_word), n_word)
  nature <- unlist(sapply(seg_word, names))
  result <- data.frame(index, unlist(seg_word), nature)
  colnames(result) <- c("id", "word","nature")
  n_word <- sapply(split(result,result$id), nrow)
  index_word <- sapply(n_word, seq_len)
  index_word <- unlist(index_word)
  result$index_word <- index_word
  return(result)
}

vec <- c("电视","小米","不错","安装")

ps_seg = return_segment(df_ps,vec)
pos_term = return_result(ps_seg)

return_freq <- function(result,is_sort = TRUE){
  word.frep <- table(result$word)
  
  if(is_sort){
    word.frep <- sort(word.frep, decreasing = TRUE)
  }
  
  word.frep <- data.frame(word.frep)
  #word.frep <- word.frep[!is.na(word.frep$Var1),]
  #word.frep <- word.frep[which(word.frep$Var1 != "NA"),]
  #word.frep <- word.frep[which(word.frep$Var1 != "日期"),]
  #word.frep <- word.frep[which(word.frep$Var1 != "矮"),]
  return (word.frep)
}

freq = return_freq(pos_term, is_sort = FALSE)
colnames(freq) <- c('word','frequency')

dfbind <- merge(pos_term, freq, by='word', all = TRUE)
dfbind <- dfbind[order(dfbind$id, decreasing = FALSE),]


wordcloud2(freq[0:100,],
           color = "random-dark",
           size = 1.3,minSize = 0.7,
           shape = "alias of square",
           rotateRatio = 0.2)


write.csv(freq, "posterm_freq.csv", row.names = FALSE)
write.csv(pos_term, "pos_term.csv", row.names = FALSE)
write.csv(dfbind, "pos_all.csv", row.names = FALSE)
