if(FALSE){
  # -*- coding: utf-8 -*-
  "
  Created on Tue Nov 26 21:27:12 2019
  @author: chenxinye
  "
}


library(magrittr) 
library(jiebaR)
library(tidytext)
library(dplyr)

setwd('C:/Users/Administrator/Desktop/E-commerce information mining')
cutter <- worker(type = "tag", stop_word = "dict/stoplist.txt")

SAVE = F

sku_collect = c(
  '7824307','136360','7265743','46472869374','100000198663','51382064682','3567887'
)

#define the word beyond the stop words, which will be deleted later!
nonword_vec <- c("亲爱",
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
                 "在线",
                 "如有",
                 "尝试",
                 "自营",
                 "正规",
                 "安装",
                 "疑问"
)

load_data <- function(sku_collect){
  if(FALSE)
  {
    "
    In this section, I just got positve and negative datas respectively.
    To enlarge the data, you can get data by web crawler I wrote from  https://github.com/chenxinye/web-crawler-for-reviews
    "
  }
  n_len = length(sku_collect)
  pbind_reviews = data.frame();nbind_reviews = data.frame()
  
  for (i in 1:n_len){
    pfile = paste('org_data/', sku_collect[i], sep = '')
    pfile = paste(pfile, 'po.csv', sep = '')
    nfile = paste('org_data/', sku_collect[i], sep = '')
    nfile = paste(nfile, 'ne.csv', sep = '')
    nreviews = read.csv(nfile, encoding = 'UTF-8')
    nbind_reviews = rbind(nbind_reviews, nreviews)
    previews = read.csv(pfile, encoding = 'UTF-8')
    pbind_reviews = rbind(pbind_reviews, previews)
  }
  
  return(c(pbind_reviews, nbind_reviews))
}

delete_nonuseword <- function(comment,vector,numbool=T,wordbool=F){
  # delete some useless notations, numbers, etc.
  reviews = comment
  if(numbool){
    reviews = gsub("[a-zA-Z0-9]", "", reviews)
  }
  
  if(wordbool){
    for (i in vector){reviews = gsub(as.character(i), "", reviews)}
  }
  
  reviews = gsub("[&|;]", "", reviews)
  reviews = gsub("\\\\", "", reviews)
  return (reviews)
}


return_segment <- function(df,vector,nb=T,wb=F){
  reviews <- delete_nonuseword(df,vector,numbool=nb, wordbool=wb)
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


reviews <- sku_collect %>% load_data()
previews <- reviews[1]; nreviews <- reviews[2]



delete_useless = FALSE # do not use nonword_vec.

if (delete_useless){
  ps_seg = return_segment(previews$X.U.FEFF.comments,nonword_vec,nb=T,wb=T)
  ng_seg = return_segment(nreviews$X.U.FEFF.comments,nonword_vec,nb=T,wb=T)
}else{
  ps_seg <- return_segment(previews$X.U.FEFF.comments,nonword_vec,nb=T,wb=F)
  ng_seg <- return_segment(nreviews$X.U.FEFF.comments,nonword_vec,nb=T,wb=F)
}

ps_term = ps_seg %>% return_result()
ng_term = ng_seg %>% return_result()


tfidf_Get <- function(term){
  #In this part, I get tf-idf of each word.
  term$word %>%table() -> frequency
  data.frame(frequency %>% sort(decreasing = TRUE)) -> frequency
  c('word', 'n') -> names(frequency)
  
  inner_join(ps_term, frequency) -> term
  term %>% bind_tf_idf(word, id, n) %>% arrange(desc(tf_idf)) -> return_df
  print(head(return_df, 20))
  return(return_df)
}

ps_term %>% tfidf_Get() -> pstfidf; ng_term %>% tfidf_Get() -> ngtfidf

print('useless words of positive words')
pstfidf[which(pstfidf$n < 50 & pstfidf$tf_idf < 0.0004),] -> Ps_nonword_vec
tail(Ps_nonword_vec, 125)

print('useless words of negative words') # "Brushing" word/conversation 
ngtfidf[which(ngtfidf$n < 50 & ngtfidf$tf_idf < 0.0005),] -> Ng_nonword_vec
tail(Ng_nonword_vec, 125)


#just test!
Ps_nonword_vec$id -> Ps_conversation_id
previews$X.U.FEFF.comments[Ps_conversation_id] -> Ps_conversation

Ng_nonword_vec$id -> Ng_conversation_id
nreviews$X.U.FEFF.comments[Ng_conversation_id] -> Ng_conversation

#get results
if(SAVE){
  pstfidf %>% write.csv(file = "save/pstfidf.csv", row.names = FALSE)
  ngtfidf %>% write.csv(file = "save/ngtfidf.csv", row.names = FALSE)
  
}

