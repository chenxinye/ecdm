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

setwd('I:/E-commerce information mining/history')
df_ng <- read.csv("100003142993ne.csv", header=F, encoding = 'UTF-8')


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



vec <- c("小米",
         "电视",
         "亲爱",
         "官方",
         "客服",
         "竭诚",
         "咨询",
         "愉快",
         "京东",
         "服务",
         "反馈",
         "支持",
         "热线",
         "致电",
         "体验",
         "产品",
         "放心使用",
         "出厂",
         "享受",
         "在线",
         "如有",
         "尝试",
         "更快",
         "自营",
         "正规",
         "安装",
         "疑问",
         "米粉")
#差评会话有大部分客服话语的干扰，去掉常用谦辞


ng_seg = return_segment(df_ng,vec)
neg_term = return_result(ng_seg)


return_freq <- function(result,is_sort = TRUE){
  word.frep <- table(result$word)
  
  if(is_sort){
    word.frep <- sort(word.frep, decreasing = TRUE)
  }
  
  word.frep <- data.frame(word.frep)
  #word.frep <- word.frep[!is.na(word.frep$Var1),]
  #word.frep <- word.frep[which(word.frep$Var1 != "NA"),]
  #word.frep <- word.frep[which(word.frep$Var1 != "日期"),]
  #word.frep <- word.frep[which(word.frep$Var1 != "�?"),]
  return (word.frep)
}


freq = return_freq(neg_term)
colnames(freq) <- c('word','frequency')


dfbind <- merge(neg_term, freq, by='word', all = TRUE)
dfbind <- dfbind[order(dfbind$id, decreasing = FALSE),]


wordcloud2(freq[0:100,],
           color = "random-dark",
           size = 0.9,
           minSize = 1,
           shape = "alias of square",
           rotateRatio=0.2)


write.csv(freq, "negterm_freq.csv", row.names = FALSE)
write.csv(neg_term, "neg_term.csv", row.names = FALSE)
write.csv(dfbind, "neg_all.csv", row.names = FALSE)
