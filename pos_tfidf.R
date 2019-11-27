if(FALSE){
  # -*- coding: utf-8 -*-
  "
  Created on Tue Nov 26 21:27:12 2019
  @author: chenxinye
  "
}

library(pacman)
p_load(jiebaR,wordcloud2,tidyverse,tidytext,data.table,rio)

setwd('I:/E-commerce information mining/history')
df_ps <- read.csv("pos_all.csv", header=T)

df_ps %>% bind_tf_idf(
  term = word,
  document = id,
  n = frequency
  ) -> df_ps

df_ps %>% unnest() %>% count(id,word) -> f_table

filter_word <- c('米粉',
                 '我家',
                 '爱',
                 '第二次',
                 '未填写',
                 '做个',
                 '总体',
                 '方法'
                 )

tf_idf <- f_table %>% bind_tf_idf(
                  term = word,
                  document = id,
                  n = n
                  )

for (i in filter_word){
  tf_idf <- tf_idf[which(tf_idf$word != i),]
}
  
top10 <- tf_idf %>% group_by(id) %>% top_n(10,tf_idf)
top10 <- top10 %>% ungroup() 


worditem <- data.frame(top10$word,top10$tf_idf)
worditem <- worditem[order(worditem$top10.tf_idf, decreasing = T),]


worditem <- worditem[!duplicated(worditem$top10.word),]
worditem_get <- head(worditem,100)

worditem_get %>% wordcloud2(color = "random-dark",
                        size = 0.6,
                        minSize = 0.2,
                        shape = "alias of square",
                        rotateRatio=0.2)

write.csv(worditem, "pos_worditem.csv", row.names = FALSE)
