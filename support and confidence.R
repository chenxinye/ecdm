#@author: chenxinye

library(arules)
library(rattle)
library(arulesViz)
setwd('I:/E-commerce information mining/history')

positive_word <- read.csv("pos_term.csv")

return_summary <- function(comment_word,sup = 0.8,con = 0.1,boole){
  table_aprori <- as(split(comment_word$word,comment_word$id),"transactions")
  model_build <- apriori(table_aprori,parameter = list(support = sup,confidence=con))
  
  output <- inspect(sort(model_build,by = "confidence"))
  summary(model_build)
  output <- output[-which(output$lhs == "{}"),];head(output,30)
  if (boole == T){
    return (output)
  }else{
    return (model_build)
  }
}

return_writer <- function(df,string){
  write.csv(df, string, row.names = FALSE)}


pos <- return_summary(positive_word,sup= 0.003, con = 0.005,T)
write.csv(pos, "aprori for JD review.csv", row.names = FALSE)

pos_m <- return_summary(positive_word,sup= 0.003, con = 0.005,F)
output_pos <- head(sort(pos_m,by = "confidence"),20)
plot(output_pos,method = "grouped",measure = "lift",shading ="support")