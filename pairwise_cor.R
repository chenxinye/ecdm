if(FALSE){
  # -*- coding: utf-8 -*-
  "
  Created on Tue Nov 26 21:27:12 2019
  @author: chenxinye
  "
}
library(widyr)
library(tidyr)
library(purrr)
library(readr)

setwd('I:/E-commerce information mining')

pdf = read.csv('save/pstfidf.csv')
ndf = read.csv('save/ngtfidf.csv')

#May not be fit for the conversation id
return_cor <- function(pdf){
  count_cors <- pdf %>% pairwise_cor(id, word, n, sort = T)
  tf_idf_cors <- pdf %>% pairwise_cor(id, word, tf_idf, sort = T)
  #count_cors <- count_cors[which(count_cors$correlation != 1.),]
  #tf_idf_cors <- tf_idf_cors[which(tf_idf_cors$correlation != 1.),]
  names(count_cors) <- c("item1","item2","correlation") ;names(tf_idf_cors) <- c("item1","item2","correlation")
  return(c(count_cors,tf_idf_cors))
}

pdf %>% return_cor() -> pcor 
ndf %>% return_cor() -> ncor


sku_collect = c(
  '7824307','136360','7265743','46472869374','100000198663','51382064682','3567887'
)

cutter <- worker(type = "tag", stop_word = "dict/stoplist.txt")

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


load_data <- function(sku, properties = 'po'){
  if(FALSE)
  {
    "
    In this section, I just got positve and negative datas respectively.
    To enlarge the data, you can get data by web crawler I wrote from  https://github.com/chenxinye/web-crawler-for-reviews
    "
  }
  reviews = data.frame()
  if(properties == 'po'){
    tryCatch({
      file = paste('org_data/', sku, sep = '')
      file = paste(file, 'po.csv', sep = '')
      reviews = read.csv(file, encoding = 'UTF-8')
      #reviews$reviews_id = rep(sku_collect,times = length(reviews$X.U.FEFF.comments))
    }, warning = function(w) {
      print(w)
    }, error = function(e) {
      print(e)
    }, finally = {
      print(paste(sku, ' pass!', sep = ''))
    })
    
    return(reviews)
    
  }else if(properties == 'ne'){
    tryCatch({
      file = paste('org_data/', sku, sep = '')
      file = paste(file, 'ne.csv', sep = '')
      reviews = read.csv(file, encoding = 'UTF-8')
      #reviews$reviews_id = rep(sku_collect,times = length(reviews$X.U.FEFF.comments))
    }, warning = function(w) {
      print(w)
    }, error = function(e) {
      print(e)
    }, finally = {
      print(paste(sku, ' pass!', sep = ''))
    })
    
    return(reviews)
    
  }
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

reviews_sum <- data.frame()
for (sku in sku_collect){
  previews <- sku %>% load_data(properties = 'po');nreviews <- sku %>% load_data(properties = 'ne')
  delete_useless = FALSE # do not use nonword_vec.
  
  tryCatch({
    if (delete_useless){
      ps_seg <- return_segment(previews$X.U.FEFF.comments,nonword_vec,nb=T,wb=T)
      ng_seg <- return_segment(nreviews$X.U.FEFF.comments,nonword_vec,nb=T,wb=T)
    }else{
      ps_seg <- return_segment(previews$X.U.FEFF.comments,nonword_vec,nb=T,wb=F)
      ng_seg <- return_segment(nreviews$X.U.FEFF.comments,nonword_vec,nb=T,wb=F)
    }
    ps_term <- ps_seg %>% return_result();ng_term <- ng_seg %>% return_result()
    
    ps_term$SKU = rep(sku,times = length(ps_term$word))
    ng_term$SKU = rep(sku,times = length(ng_term$word))
    
    ps_term$word %>% table() -> frequency
    data.frame(frequency %>% sort(decreasing = TRUE)) -> frequency
    c('word', 'n') -> names(frequency)
    inner_join(ps_term, frequency) ->  ps_term
    
    ng_term$word %>% table() -> frequency
    data.frame(frequency %>% sort(decreasing = TRUE)) -> frequency
    c('word', 'n') -> names(frequency)
    inner_join(ng_term, frequency) ->  ng_term
    
    termget = rbind(ps_term, ng_term)
    reviews_sum = rbind(reviews_sum, termget)
    
    }, error = function(e) {
      print(e)
      
    }, finally = {
      print(paste(sku, ' pass!', sep = ''))
    }
    
    )
}

return_cor <- function(reviews){
  count_cors <- reviews %>% pairwise_cor(SKU, word, n, sort = T)
  names(count_cors) <- c("SKU1","SKU2","correlation")
  return(count_cors)
}

reviews_sum %>% return_cor() -> pcor 

SAVE = TRUE
if(SAVE){
  pcor %>% write.csv(file = "save/sku_reviews_cor.csv", row.names = FALSE)
}

