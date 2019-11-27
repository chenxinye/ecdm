#@author: chenxinye

library(wordcloud2)
library(magrittr)
library(readxl)
library(jiebaR)
library(wordcloud2)
library(magrittr)

setwd('I:/E-commerce information mining/history')
df_ps <- read.csv("8764069 question answer.csv", header=F, encoding = 'UTF-8')

cutter <- worker(type = "tag", stop_word = "./dict/stoplist.txt")

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
  return(seg_word)}

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

vec <- c('好吗', '真的', '奶粉', '月', '水','孩子',
         '喝','段','亲们','问','启赋','宝宝','吃',
         '想','甜','这款','我家','里','三个','宝妈们',
         '店','好像','新','勺')

ps_seg = return_segment(df_ps,vec)
pos_term = return_result(ps_seg)

return_freq <- function(result){
  word.frep <- table(result$word)
  word.frep <- sort(word.frep, decreasing = TRUE)
  word.frep <- data.frame(word.frep)
  word.frep <- word.frep[!is.na(word.frep$Var1),]
  word.frep <- word.frep[which(word.frep$Var1 != "NA"),]
  word.frep <- word.frep[which(word.frep$Var1 != "日期"),]
  #word.frep <- word.frep[which(word.frep$Var1 != "矮"),]
  return (word.frep)
}

freq = return_freq(pos_term)
wordcloud2(freq[0:100,],color = "random-dark",size = 0.7,minSize = 0.2,shape = "alias of square",rotateRatio=0.2)
write.csv(freq, "term_freq.csv", row.names = FALSE)
write.csv(pos_term, "pos_term.csv", row.names = FALSE)

