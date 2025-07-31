# Comprehensive R Programming Reference Guide

## Table of Contents
1. [Basic R Components](#basic-r-components)
2. [Data Structures](#data-structures)
3. [Data Import/Export](#data-importexport)
4. [Data Manipulation](#data-manipulation)
5. [Data Visualization](#data-visualization)
6. [Statistical Analysis](#statistical-analysis)
7. [Programming Techniques](#programming-techniques)
8. [Package Development](#package-development)
9. [Performance Optimization](#performance-optimization)
10. [Best Practices](#best-practices)

## Basic R Components

### Environment Setup
```r
# Install packages
install.packages("package_name")
install.packages(c("dplyr", "ggplot2", "tidyr"))

# Load packages
library(package_name)
require(package_name)  # Alternative that returns TRUE/FALSE

# Update packages
update.packages()

# Check working directory
getwd()
setwd("path/to/directory")

# List objects in environment
ls()
rm(object_name)  # Remove an object
rm(list = ls())  # Clear environment
```

### Basic Operations
```r
# Arithmetic
2 + 3    # Addition
5 - 2    # Subtraction
4 * 5    # Multiplication
10 / 2   # Division
2 ^ 3    # Exponentiation
10 %% 3  # Modulo (remainder)
10 %/% 3 # Integer division

# Assignment
x <- 5         # Preferred
x = 5          # Also works
5 -> x         # Reverse assignment
assign("x", 5) # Function-based assignment

# Logical operators
a & b    # AND (vectorized)
a && b   # AND (single value)
a | b    # OR (vectorized) 
a || b   # OR (single value)
!a       # NOT
xor(a,b) # Exclusive OR

# Comparison
a == b   # Equal to
a != b   # Not equal to
a > b    # Greater than
a >= b   # Greater than or equal
a < b    # Less than
a <= b   # Less than or equal
```

### R Help
```r
# Get help for a function
?function_name
help(function_name)

# Search for a keyword
??keyword
help.search("keyword")

# View function source code
getAnywhere(function_name)

# View function arguments
args(function_name)

# View vignettes
vignette()
vignette("package_name")
```

## Data Structures

### Vectors
```r
# Create vectors
x <- c(1, 2, 3, 4, 5)              # Numeric vector
y <- c("a", "b", "c")              # Character vector
z <- c(TRUE, FALSE, TRUE)          # Logical vector

# Sequence vectors
1:10                               # Integer sequence
seq(1, 10, by = 2)                 # Sequence with step
seq(1, 10, length.out = 5)         # Sequence with length
rep(1, times = 5)                  # Repeat value
rep(c(1, 2), each = 3)             # Repeat elements

# Vector operations
length(x)                          # Vector length
x[3]                               # Access element
x[c(1, 3, 5)]                      # Multiple elements
x[-2]                              # Exclude element
x[x > 3]                           # Conditional selection
```

### Matrices
```r
# Create matrices
matrix(1:9, nrow = 3, ncol = 3)          # By column (default)
matrix(1:9, nrow = 3, ncol = 3, byrow = TRUE)  # Fill by row

# Matrix operations
rbind(x, y)                              # Bind rows
cbind(x, y)                              # Bind columns
t(m)                                     # Transpose
m1 %*% m2                                # Matrix multiplication
solve(m)                                 # Matrix inverse
eigen(m)                                 # Eigenvalues and vectors
diag(m)                                  # Diagonal elements
diag(3)                                  # Identity matrix

# Accessing elements
m[2, 3]                                  # Row 2, column 3
m[2, ]                                   # Row 2
m[, 3]                                   # Column 3
m[m > 5]                                 # Conditional selection
```

### Lists
```r
# Create lists
my_list <- list(a = 1:3, b = "hello", c = TRUE, d = function(x) x^2)

# Accessing elements
my_list$a                             # By name
my_list[["a"]]                        # By name (alternative)
my_list[[1]]                          # By position
my_list[1:2]                          # Sublist (multiple elements)

# List operations
length(my_list)                       # Number of elements
names(my_list)                        # Element names
unlist(my_list)                       # Convert to vector
lapply(my_list, function(x) x)        # Apply function to each element
```

### Data Frames
```r
# Create data frames
df <- data.frame(
  id = 1:3,
  name = c("Alice", "Bob", "Charlie"),
  score = c(95, 82, 91),
  stringsAsFactors = FALSE  # Don't convert strings to factors
)

# Accessing elements
df$name                       # Column by name
df[["name"]]                  # Column by name (alternative)
df[, "name"]                  # Column by name (alternative)
df[2, ]                       # Row by position
df[2, 3]                      # Specific cell
df[df$score > 90, ]           # Conditional selection

# Data frame operations
nrow(df)                      # Number of rows
ncol(df)                      # Number of columns
dim(df)                       # Dimensions
str(df)                       # Structure
summary(df)                   # Summary statistics
head(df, n = 2)               # First n rows
tail(df, n = 2)               # Last n rows
colnames(df)                  # Column names
rownames(df)                  # Row names
```

### Factors
```r
# Create factors
gender <- factor(c("male", "female", "male", "female"))
ordered_factor <- factor(c("low", "medium", "high"), 
                         levels = c("low", "medium", "high"), 
                         ordered = TRUE)

# Factor operations
levels(gender)                # Factor levels
nlevels(gender)               # Number of levels
as.numeric(ordered_factor)    # Convert to numeric
```

### Arrays
```r
# Create arrays
arr <- array(1:24, dim = c(2, 3, 4))  # 2 rows, 3 columns, 4 matrices

# Access elements
arr[1, 2, 3]                          # Specific element
arr[, , 2]                            # Specific matrix
```

## Data Import/Export

### CSV Files
```r
# Import CSV
df <- read.csv("file.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)
df <- read.csv2("file.csv", header = TRUE, sep = ";")  # European format

# Export CSV
write.csv(df, "output.csv", row.names = FALSE)
write.csv2(df, "output.csv", row.names = FALSE)  # European format
```

### Excel Files
```r
# Install and load readxl package
install.packages("readxl")
library(readxl)

# Import Excel
df <- read_excel("file.xlsx", sheet = 1)
df <- read_excel("file.xlsx", sheet = "Sheet1")

# Export Excel (using writexl)
library(writexl)
write_xlsx(df, "output.xlsx")
```

### Other Formats
```r
# RData/RDS files
save(obj1, obj2, file = "objects.RData")  # Save multiple objects
load("objects.RData")                     # Load multiple objects
saveRDS(df, "dataframe.rds")              # Save single object
df <- readRDS("dataframe.rds")            # Load single object

# Text files
df <- read.table("file.txt", header = TRUE, sep = "\t")
write.table(df, "output.txt", sep = "\t", row.names = FALSE)

# JSON files
library(jsonlite)
df <- fromJSON("file.json")
write_json(df, "output.json", pretty = TRUE)

# Database connections
library(DBI)
library(RSQLite)
con <- dbConnect(SQLite(), "database.db")
df <- dbGetQuery(con, "SELECT * FROM table")
dbWriteTable(con, "table_name", df)
dbDisconnect(con)
```

### Web Data
```r
# Web scraping
library(rvest)
webpage <- read_html("https://example.com")
tables <- html_table(webpage)
text <- html_text(html_nodes(webpage, "p"))

# APIs
library(httr)
response <- GET("https://api.example.com/data")
content <- content(response, "parsed")
```

## Data Manipulation

### Base R
```r
# Subsetting
subset(df, score > 90)                  # Select rows with condition
df[df$score > 90, c("id", "name")]      # Select rows and columns

# Merging
merged_df <- merge(df1, df2, by = "id")  # Inner join
merged_df <- merge(df1, df2, by = "id", all = TRUE)  # Full join
merged_df <- merge(df1, df2, by = "id", all.x = TRUE)  # Left join
merged_df <- merge(df1, df2, by = "id", all.y = TRUE)  # Right join

# Aggregation
aggregate(score ~ gender, data = df, FUN = mean)
```

### Tidyverse
```r
library(tidyverse)  # Loads dplyr, tidyr, ggplot2, etc.

# dplyr basics
df %>%                           # Pipe operator
  filter(score > 90) %>%         # Filter rows
  select(id, name) %>%           # Select columns
  mutate(grade = score / 10) %>% # Create new column
  arrange(desc(score)) %>%       # Sort
  group_by(gender) %>%           # Group
  summarize(avg = mean(score))   # Summarize

# Joining with dplyr
left_join(df1, df2, by = "id")
right_join(df1, df2, by = "id")
inner_join(df1, df2, by = "id")
full_join(df1, df2, by = "id")

# tidyr basics
df %>%
  pivot_longer(cols = c(math, science), 
               names_to = "subject", 
               values_to = "score")  # Wide to long format

df %>%
  pivot_wider(names_from = subject, 
              values_from = score)   # Long to wide format

df %>% 
  separate(date, into = c("year", "month", "day"), sep = "-")  # Split column

df %>%
  fill(missing_column)  # Fill NA values
```

### data.table
```r
library(data.table)
dt <- as.data.table(df)

# Basic syntax: DT[i, j, by]
dt[score > 90]                              # Filter rows
dt[, .(name, score)]                        # Select columns
dt[, grade := score/10]                     # Add/update column
dt[, .(avg_score = mean(score)), by = gender]  # Group by and summarize
dt[order(-score)]                           # Sort

# Joining
dt1[dt2, on = "id"]                         # Join
```

## Data Visualization

### Base Graphics
```r
# Basic plots
plot(x, y)                            # Scatter plot
hist(x)                               # Histogram
boxplot(x ~ group)                    # Box plot
barplot(table(factor))                # Bar plot
pie(table(factor))                    # Pie chart
plot(density(x))                      # Density plot

# Plot customization
plot(x, y,
     main = "Title",                  # Title
     xlab = "X axis", ylab = "Y axis", # Axis labels
     col = "blue",                    # Color
     pch = 16,                        # Point type
     cex = 1.5)                       # Point size

# Multiple plots
par(mfrow = c(2, 2))                  # 2x2 grid of plots
```

### ggplot2
```r
library(ggplot2)

# Basic structure
ggplot(df, aes(x = x, y = y)) +
  geom_point()

# Common geoms
ggplot(df, aes(x = x, y = y)) +
  geom_point() +                      # Scatter plot
  geom_line() +                       # Line
  geom_smooth(method = "lm") +        # Trend line
  geom_text(aes(label = label))       # Text labels

ggplot(df, aes(x = x)) +
  geom_histogram(bins = 30)           # Histogram

ggplot(df, aes(x = factor, y = y)) +
  geom_boxplot()                      # Box plot

ggplot(df, aes(x = factor, fill = group)) +
  geom_bar(position = "dodge")        # Grouped bar plot

# Faceting
ggplot(df, aes(x = x, y = y)) +
  geom_point() +
  facet_wrap(~ group)                 # Split by one variable
  
ggplot(df, aes(x = x, y = y)) +
  geom_point() +
  facet_grid(group1 ~ group2)         # Split by two variables

# Themes and customization
ggplot(df, aes(x = x, y = y)) +
  geom_point() +
  labs(title = "Title",
       x = "X axis",
       y = "Y axis") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45),
    legend.position = "bottom"
  )
```

### Interactive Plots
```r
# plotly
library(plotly)
p <- ggplot(df, aes(x = x, y = y, color = group)) + geom_point()
ggplotly(p)

# Direct with plotly
plot_ly(df, x = ~x, y = ~y, color = ~group, type = "scatter", mode = "markers")

# htmlwidgets
library(highcharter)
highchart() %>%
  hc_add_series(df, "scatter", hcaes(x = x, y = y, group = group))
```

## Statistical Analysis

### Descriptive Statistics
```r
# Summary statistics
mean(x)                     # Mean
median(x)                   # Median
sd(x)                       # Standard deviation
var(x)                      # Variance
min(x)                      # Minimum
max(x)                      # Maximum
range(x)                    # Range
quantile(x)                 # Quantiles
IQR(x)                      # Interquartile range
summary(x)                  # Summary statistics
cor(x, y)                   # Correlation
cov(x, y)                   # Covariance
```

### Statistical Tests
```r
# t-tests
t.test(x, y)                # Two-sample t-test
t.test(x, y, paired = TRUE) # Paired t-test
t.test(x, mu = 0)           # One-sample t-test

# ANOVA
aov_result <- aov(y ~ group, data = df)
summary(aov_result)
TukeyHSD(aov_result)        # Post-hoc test

# Non-parametric tests
wilcox.test(x, y)           # Wilcoxon test
kruskal.test(y ~ group, data = df)  # Kruskal-Wallis test

# Chi-squared test
chisq.test(table(factor1, factor2))

# Correlation tests
cor.test(x, y)              # Correlation test
```

### Regression
```r
# Linear regression
model <- lm(y ~ x1 + x2, data = df)
summary(model)              # Model summary
coef(model)                 # Coefficients
fitted(model)               # Fitted values
residuals(model)            # Residuals
predict(model, newdata)     # Predictions
confint(model)              # Confidence intervals
anova(model)                # ANOVA table

# Generalized linear models
glm_model <- glm(y ~ x1 + x2, family = binomial, data = df)  # Logistic regression
glm_model <- glm(y ~ x1 + x2, family = poisson, data = df)   # Poisson regression

# Mixed effects models
library(lme4)
mixed_model <- lmer(y ~ x + (1|group), data = df)
summary(mixed_model)
```

### Machine Learning
```r
# Data splitting
library(caret)
set.seed(123)
trainIndex <- createDataPartition(df$y, p = 0.8, list = FALSE)
train_data <- df[trainIndex, ]
test_data <- df[-trainIndex, ]

# Cross-validation
ctrl <- trainControl(method = "cv", number = 10)
model <- train(y ~ ., data = train_data, method = "rf", trControl = ctrl)

# Random Forest
library(randomForest)
rf_model <- randomForest(y ~ ., data = train_data)
importance(rf_model)
predict(rf_model, test_data)

# Support Vector Machine
library(e1071)
svm_model <- svm(y ~ ., data = train_data)
predict(svm_model, test_data)

# K-means clustering
kmeans_result <- kmeans(df[, c("x1", "x2")], centers = 3)
df$cluster <- kmeans_result$cluster
```

## Programming Techniques

### Control Structures
```r
# Conditional statements
if (condition) {
  # code
} else if (another_condition) {
  # code
} else {
  # code
}

# Switch
switch(x,
  "a" = 1,
  "b" = 2,
  3  # default
)

# Loops
for (i in 1:10) {
  # code
}

while (condition) {
  # code
}

repeat {
  # code
  if (condition) break
}
```

### Functions
```r
# Basic function
my_function <- function(arg1, arg2 = default_value) {
  result <- arg1 + arg2
  return(result)
}

# Ellipsis (...) argument
my_function <- function(arg1, ...) {
  other_func(arg1, ...)
}

# Anonymous functions
lapply(list, function(x) x^2)

# Closures
make_multiplier <- function(n) {
  function(x) x * n
}
double <- make_multiplier(2)
double(5)  # Returns 10
```

### Functional Programming
```r
# apply family
apply(matrix, 1, mean)       # Apply function to rows (1) or columns (2)
lapply(list, function)       # Apply to list, return list
sapply(list, function)       # Apply to list, simplify result
tapply(vector, factor, function)  # Apply by factor levels
mapply(function, list1, list2)    # Apply to multiple lists

# purrr package
library(purrr)
map(list, function)          # Like lapply
map_dbl(list, function)      # Return numeric vector
map2(list1, list2, function) # Map over two lists
pmap(list(list1, list2), function)  # Map over multiple lists
reduce(list, function)       # Reduce list to single value
```

### Error Handling
```r
# Try-catch
result <- tryCatch(
  {
    # code that might fail
    1 + "a"
  },
  error = function(e) {
    # error handler
    message("An error occurred: ", e$message)
    return(NA)
  },
  warning = function(w) {
    # warning handler
    message("A warning occurred: ", w$message)
    return(NULL)
  },
  finally = {
    # cleanup code
    message("This runs regardless of success or failure")
  }
)

# Simpler version
try(expression, silent = TRUE)
```

### S3 and S4 Classes
```r
# S3 class
create_person <- function(name, age) {
  person <- list(name = name, age = age)
  class(person) <- "person"
  return(person)
}

# S3 method
print.person <- function(x, ...) {
  cat("Person: ", x$name, ", Age: ", x$age, "\n", sep = "")
}

# S4 class
setClass("Person",
  slots = c(
    name = "character",
    age = "numeric"
  )
)

# S4 method
setMethod("show",
  signature = "Person",
  definition = function(object) {
    cat("Person: ", object@name, ", Age: ", object@age, "\n", sep = "")
  }
)

# Create S4 object
new_person <- new("Person", name = "Alice", age = 30)
```

## Package Development

### Package Structure
```
mypackage/
├── DESCRIPTION      # Package metadata
├── NAMESPACE        # Export and import declarations
├── R/               # R code
├── man/             # Documentation
├── data/            # Data files
├── src/             # C/C++/Fortran code
├── tests/           # Tests
├── vignettes/       # Long-form documentation
└── inst/            # Additional files
```

### Creating a Package
```r
# Install required tools
install.packages(c("devtools", "roxygen2", "testthat", "knitr"))

# Create package skeleton
library(devtools)
create_package("mypackage")
use_r("functions")           # Create R file
use_data(my_data)            # Add data
use_testthat()               # Set up testing
use_vignette("introduction") # Create vignette
use_package("dplyr")         # Add dependency

# Document functions with roxygen2
#' @title Function Title
#' @description Function description
#' @param arg1 Description of arg1
#' @return Description of return value
#' @examples
#' example_function(1)
#' @export
example_function <- function(arg1) {
  # function code
}

# Generate documentation
document()

# Test package
test()

# Build and check
check()
build()

# Install locally
install()
```

## Performance Optimization

### Profiling and Benchmarking
```r
# System time
system.time(expression)

# Benchmarking
library(microbenchmark)
microbenchmark(
  method1 = expression1,
  method2 = expression2,
  times = 100
)

# Profiling
Rprof("profile.out")
# code to profile
Rprof(NULL)
summaryRprof("profile.out")

# Memory usage
object.size(obj)
memory.profile()
gc()  # Garbage collection
```

### Vectorization
```r
# Vectorized operation (fast)
x <- 1:1000000
result <- x^2

# Loop version (slow)
result <- numeric(length(x))
for (i in seq_along(x)) {
  result[i] <- x[i]^2
}

# Vectorized functions
ifelse(condition, yes, no)
pmin(x, y)  # Parallel min
pmax(x, y)  # Parallel max
```

### Parallel Processing
```r
# Basic parallel processing
library(parallel)
n_cores <- detectCores() - 1

# Parallel lapply
cl <- makeCluster(n_cores)
clusterExport(cl, c("data", "function_name"))
results <- parLapply(cl, X, function_name)
stopCluster(cl)

# Parallel processing with foreach
library(foreach)
library(doParallel)
registerDoParallel(n_cores)

results <- foreach(i = 1:10, .combine = rbind) %dopar% {
  # code to run in parallel
}
```

### C++ Integration with Rcpp
```r
# Install Rcpp
install.packages("Rcpp")
library(Rcpp)

# Inline C++ code
cppFunction('
int fibonacci(int n) {
  if (n < 2) return n;
  return fibonacci(n-1) + fibonacci(n-2);
}
')

# Use the C++ function
fibonacci(10)

# Create C++ source file
sourceCpp("file.cpp")
```

## Best Practices

### Code Style
```r
# Variable and function names: use snake_case
my_variable <- 1
my_function <- function() {}

# Constants: use UPPERCASE
MAX_ITERATIONS <- 1000

# Class names: use PascalCase
MyClass <- setClass("MyClass")

# Indentation: 2 spaces
if (condition) {
  do_something()
}

# Line length: maximum 80 characters

# Function documentation: use roxygen2
```

### Package Management
```r
# Use renv for package management
install.packages("renv")
renv::init()        # Initialize project
renv::snapshot()    # Save package state
renv::restore()     # Restore package state

# Check package dependencies
packageVersion("dplyr")
```

### Version Control
```r
# Use git with R projects
library(usethis)
use_git()
use_github()

# Add .gitignore
use_git_ignore("*.RData")
```

### Testing
```r
# testthat tests
library(testthat)

test_that("addition works", {
  expect_equal(2 + 2, 4)
  expect_gt(5, 3)
  expect_error(1 + "a")
})

# Run all tests
test_dir("tests/testthat")
```

### Documentation
```r
# Roxygen2 documentation
#' Title
#'
#' @param x Description of parameter
#' @return Description of return value
#' @examples
#' example_function(5)
#' @export
example_function <- function(x) {
  return(x * 2)
}

# Create vignettes
library(knitr)
browseVignettes()
```

### Reproducibility
```r
# Set random seed
set.seed(123)

# Session info
sessionInfo()

# Save/load workspace
save.image("workspace.RData")
load("workspace.RData")

# Use RMarkdown for reproducible reports
library(rmarkdown)
render("report.Rmd", "pdf_document")
```

### Efficient R Tips
1. Preallocate memory for data structures
2. Use specialized data structures (data.table for large data)
3. Vectorize operations when possible
4. Use appropriate apply functions instead of loops
5. Profile code to identify bottlenecks
6. Keep objects in memory to avoid repeated I/O
7. Use native R functions when available
8. Consider Rcpp for performance-critical code
9. Store data in binary formats (.rds, .RData) for faster I/O
10. Use proper subsetting techniques



# R dplyr Comprehensive Reference Card

## Overview

dplyr is a grammar of data manipulation in R, providing a consistent set of verbs that help you solve the most common data manipulation challenges. It's part of the tidyverse ecosystem.

```r
# Install and load
install.packages("dplyr")
library(dplyr)
```

## Basic Verbs

### `filter()`: Subset rows based on conditions

```r
# Filter rows where value is greater than 5
df %>% filter(value > 5)

# Multiple conditions (AND)
df %>% filter(value > 5, category == "A")

# OR condition
df %>% filter(value > 5 | category == "A")

# NOT condition
df %>% filter(!(category == "A"))
df %>% filter(category != "A")

# Working with NA values
df %>% filter(!is.na(value))
```

### `select()`: Subset columns by name

```r
# Select specific columns
df %>% select(id, name, value)

# Select columns by position
df %>% select(1, 3, 5)

# Select all columns between id and value (inclusive)
df %>% select(id:value)

# Select all columns except certain ones
df %>% select(-category)
df %>% select(-c(category, date))

# Select columns that match patterns
df %>% select(starts_with("cat"))
df %>% select(ends_with("_id"))
df %>% select(contains("value"))
df %>% select(matches(".*_\\d{2}"))  # regex pattern
```

### `mutate()`: Create or transform variables

```r
# Create new column
df %>% mutate(new_col = value * 2)

# Create multiple columns
df %>% mutate(
  ratio = x / y,
  pct = ratio * 100
)

# Conditional transformations
df %>% mutate(
  category_group = case_when(
    category %in% c("A", "B") ~ "Group 1",
    category %in% c("C", "D") ~ "Group 2",
    TRUE ~ "Other"  # default case
  )
)

# Replace existing columns
df %>% mutate(value = ifelse(value < 0, 0, value))
```

### `arrange()`: Sort rows

```r
# Sort by a column (ascending)
df %>% arrange(date)

# Sort by a column (descending)
df %>% arrange(desc(value))

# Sort by multiple columns
df %>% arrange(category, desc(value))
```

### `summarize()` / `summarise()`: Reduce multiple values to a single value

```r
# Basic summary statistics
df %>% summarize(
  mean_value = mean(value, na.rm = TRUE),
  median_value = median(value, na.rm = TRUE),
  min_value = min(value, na.rm = TRUE),
  max_value = max(value, na.rm = TRUE),
  count = n(),
  count_unique = n_distinct(category)
)
```

### `group_by()`: Group data by variables

```r
# Group by one variable
df %>%
  group_by(category) %>%
  summarize(
    avg_value = mean(value, na.rm = TRUE),
    count = n()
  )

# Group by multiple variables
df %>%
  group_by(category, year) %>%
  summarize(avg_value = mean(value, na.rm = TRUE))

# Remove grouping
df %>%
  group_by(category) %>%
  summarize(avg_value = mean(value, na.rm = TRUE)) %>%
  ungroup()
```

## Intermediate Functions

### `rename()`: Rename columns

```r
# Rename one column
df %>% rename(new_name = old_name)

# Rename multiple columns
df %>% rename(
  new_name1 = old_name1,
  new_name2 = old_name2
)
```

### `distinct()`: Return unique rows

```r
# Get distinct combinations of specified columns
df %>% distinct(category, subcategory)

# Keep all columns 
df %>% distinct(category, subcategory, .keep_all = TRUE)
```

### `count()` and `tally()`: Count observations by group

```r
# Count number of rows
df %>% count()

# Count by variable
df %>% count(category)

# Count by multiple variables with sorting
df %>% count(category, year, sort = TRUE)

# Add weights
df %>% count(category, wt = value)

# tally() is similar but works with already grouped data
df %>% 
  group_by(category) %>% 
  tally()
```

### `slice()`: Extract rows by position

```r
# Select rows by position
df %>% slice(1:5)

# Select last 5 rows
df %>% slice(n()-4:n())

# Select first row from each group
df %>% 
  group_by(category) %>%
  slice(1)
```

### Specialized slice functions

```r
# Select rows with min/max values
df %>% slice_min(value, n = 5)  # 5 rows with minimum values
df %>% slice_max(value, n = 5)  # 5 rows with maximum values

# Sample rows
df %>% slice_sample(n = 10)     # 10 random rows
df %>% slice_sample(prop = 0.1) # 10% of rows

# First/last rows per group
df %>% 
  group_by(category) %>%
  slice_head(n = 2)  # First 2 rows of each group

df %>% 
  group_by(category) %>%
  slice_tail(n = 2)  # Last 2 rows of each group
```

## Advanced Techniques

### Row-wise operations

```r
# Apply functions across rows
df %>%
  rowwise() %>%
  mutate(row_mean = mean(c(val1, val2, val3), na.rm = TRUE))

# Row-wise list-columns
df %>%
  rowwise() %>%
  mutate(
    values_list = list(c(val1, val2, val3)),
    stats = list(summary(c(val1, val2, val3)))
  )
```

### Window functions

```r
# Ranking
df %>% mutate(
  rank_simple = rank(value),
  rank_dense = dense_rank(value),
  rank_min = min_rank(value),
  rank_percent = percent_rank(value),
  rank_ntile = ntile(value, 10)  # deciles
)

# Lead and lag
df %>% 
  arrange(date) %>% 
  mutate(
    previous_value = lag(value, default = 0),
    next_value = lead(value, default = 0),
    change = value - lag(value)
  )

# Cumulative calculations
df %>% 
  arrange(date) %>% 
  mutate(
    cumulative_sum = cumsum(value),
    cumulative_mean = cummean(value),
    cumulative_min = cummin(value),
    cumulative_max = cummax(value)
  )

# Ranking within groups
df %>%
  group_by(category) %>%
  mutate(
    value_rank = min_rank(desc(value)),  # highest value = rank 1
    is_top3 = value_rank <= 3
  )
```

### Joins

```r
# Inner join (keep matches only)
inner_join(df1, df2, by = "id")

# Left join (keep all rows from df1)
left_join(df1, df2, by = "id")

# Right join (keep all rows from df2)
right_join(df1, df2, by = "id")

# Full join (keep all rows from both)
full_join(df1, df2, by = "id")

# Semi-join (filter df1 for matches in df2, keep only df1 columns)
semi_join(df1, df2, by = "id")

# Anti-join (filter df1 for non-matches in df2, keep only df1 columns)
anti_join(df1, df2, by = "id")

# Join on different column names
left_join(df1, df2, by = c("df1_id" = "df2_id"))

# Join on multiple columns
left_join(df1, df2, by = c("id", "category"))
```

### Working with databases using dplyr

```r
# Connect to a database
library(DBI)
con <- dbConnect(RSQLite::SQLite(), path = "my_database.sqlite")

# Create a reference to a table
my_table <- tbl(con, "my_table")

# Perform operations (executed lazily)
result <- my_table %>%
  filter(value > 100) %>%
  group_by(category) %>%
  summarize(avg_value = mean(value, na.rm = TRUE))

# See generated SQL
show_query(result)

# Execute and bring data into R
result_df <- result %>% collect()

# Close connection
dbDisconnect(con)
```

### Using across() for multiple columns

```r
# Apply same function to multiple columns
df %>%
  mutate(across(c(val1, val2, val3), ~ . * 2))

# Apply same function to columns matching a pattern
df %>%
  mutate(across(starts_with("val"), ~ . * 2))

# Apply same function to columns of specific types
df %>%
  mutate(across(where(is.numeric), ~ round(., 2)))

# Apply multiple functions
df %>%
  summarize(across(
    where(is.numeric),
    list(mean = mean, median = median),
    na.rm = TRUE
  ))

# Name output columns
df %>%
  summarize(across(
    c(val1, val2),
    list(avg = ~ mean(., na.rm = TRUE), 
         sd = ~ sd(., na.rm = TRUE)),
    .names = "{.col}_{.fn}"
  ))
```

### Using if_else() and case_when()

```r
# Simple conditional
df %>% mutate(
  flag = if_else(value > 100, "High", "Low", missing = "Unknown")
)

# Multiple conditions
df %>% mutate(
  category = case_when(
    value < 0 ~ "Negative",
    value >= 0 & value <= 100 ~ "Low",
    value > 100 & value <= 200 ~ "Medium",
    value > 200 ~ "High",
    TRUE ~ "Other"  # default case
  )
)
```

## Best Practices

### Data Pipeline Construction

1. **Chain operations with the pipe operator**
   ```r
   df %>%
     filter(!is.na(value)) %>%
     group_by(category) %>%
     summarize(avg = mean(value)) %>%
     arrange(desc(avg))
   ```

2. **Create intermediate variables for complex pipelines**
   ```r
   # For complex workflows, save intermediate results
   df_filtered <- df %>% filter(!is.na(value))
   df_summarized <- df_filtered %>% 
     group_by(category) %>%
     summarize(avg = mean(value))
   ```

3. **Use line breaks and indentation for clarity**
   ```r
   df %>%
     filter(
       category %in% c("A", "B", "C"),
       value > 0,
       !is.na(date)
     ) %>%
     group_by(category, year = lubridate::year(date)) %>%
     summarize(
       count = n(),
       avg_value = mean(value),
       .groups = "drop"
     )
   ```

### Performance Considerations

1. **Filter early to reduce data size**
   ```r
   # Good - filters first
   df %>%
     filter(date >= "2020-01-01") %>%
     group_by(category) %>%
     summarize(avg = mean(value))
   
   # Less efficient - processes all data first
   df %>%
     group_by(category) %>%
     summarize(avg = mean(value)) %>%
     filter(avg > 100)
   ```

2. **Use .groups argument in summarize()**
   ```r
   # Explicitly handle grouping
   df %>%
     group_by(category, subcategory) %>%
     summarize(
       avg = mean(value, na.rm = TRUE),
       .groups = "drop"  # Removes all grouping
     )
   
   # Or keep the first level of grouping
   df %>%
     group_by(category, subcategory) %>%
     summarize(
       avg = mean(value, na.rm = TRUE),
       .groups = "drop_last"  # Keeps category grouping
     )
   ```

3. **Pre-compute common values**
   ```r
   # More efficient
   df %>%
     mutate(
       ratio = value / total,
       pct = ratio * 100,  # Uses computed ratio
       grade = case_when(
         pct >= 90 ~ "A",
         pct >= 80 ~ "B",
         pct >= 70 ~ "C",
         pct >= 60 ~ "D",
         TRUE ~ "F"
       )
     )
   ```

4. **Use vectorized operations when possible**
   ```r
   # Good - vectorized
   df %>% mutate(flag = value > 100)
   
   # Avoid - row-by-row processing when not needed
   df %>% rowwise() %>% mutate(flag = value > 100)
   ```

### Miscellaneous Tips

1. **Handle NA values explicitly**
   ```r
   # Always consider NA handling
   df %>% summarize(
     avg = mean(value, na.rm = TRUE),
     count = sum(!is.na(value))
   )
   ```

2. **Use scoped verbs with `across()` for compact code**
   ```r
   # Standardize all numeric columns
   df %>%
     mutate(across(where(is.numeric), ~ scale(.)))
   ```

3. **Name summary columns clearly**
   ```r
   # Clear naming
   df %>%
     group_by(category) %>%
     summarize(
       avg_value = mean(value, na.rm = TRUE),
       med_value = median(value, na.rm = TRUE),
       count = n()
     )
   ```

4. **Use joins carefully**
   ```r
   # Check for potential many-to-many joins
   df1 %>% 
     group_by(id) %>% 
     count() %>% 
     filter(n > 1)  # Check for duplicates
   
   # Add join column identification for debugging
   left_join(df1, df2, by = "id", suffix = c("_primary", "_secondary"))
   ```

5. **Use the newer tidyverse conventions**
   ```r
   # New style
   df %>% slice_max(value, n = 5)
   
   # Old style
   df %>% top_n(5, value)
   ```


# R stringr Package Reference Card

## Introduction

The `stringr` package is part of the tidyverse and provides a cohesive set of functions for string manipulation in R. It wraps the `stringi` package, offering consistent naming patterns, function interfaces, and predictable outputs.

```r
# Installation and loading
install.packages("stringr")
library(stringr)
```

## Basic String Functions

### Creating Strings

| Function | Description | Example |
|----------|-------------|---------|
| `str_c()` | Concatenate strings | `str_c("a", "b", "c") # "abc"` |
| `str_c(sep = " ")` | Concatenate with separator | `str_c("a", "b", "c", sep = " ") # "a b c"` |
| `str_c(collapse = "")` | Collapse vector elements | `str_c(c("a", "b", "c"), collapse = "-") # "a-b-c"` |
| `str_glue()` | String interpolation | `name <- "Amy"; str_glue("Hi {name}!") # "Hi Amy!"` |
| `str_dup()` | Duplicate strings | `str_dup("abc", 2) # "abcabc"` |

### Basic String Information

| Function | Description | Example |
|----------|-------------|---------|
| `str_length()` | Count characters | `str_length("hello") # 5` |
| `str_count()` | Count pattern matches | `str_count("banana", "a") # 3` |
| `str_detect()` | Detect pattern presence | `str_detect("apple", "pl") # TRUE` |
| `str_which()` | Find indices of matches | `str_which(c("a", "b", "a"), "a") # c(1, 3)` |
| `str_locate()` | Find position of match | `str_locate("banana", "na") # matrix: c(3,4)` |
| `str_locate_all()` | Find all positions | `str_locate_all("banana", "na")` |

### String Manipulation

| Function | Description | Example |
|----------|-------------|---------|
| `str_sub()` | Extract substring | `str_sub("hello", 2, 4) # "ell"` |
| `str_sub<-()` | Replace substring | `x <- "hello"; str_sub(x, 1, 1) <- "H"; x # "Hello"` |
| `str_to_lower()` | Convert to lowercase | `str_to_lower("Hello") # "hello"` |
| `str_to_upper()` | Convert to uppercase | `str_to_upper("Hello") # "HELLO"` |
| `str_to_title()` | Convert to title case | `str_to_title("hello world") # "Hello World"` |
| `str_trim()` | Remove whitespace | `str_trim(" hello ") # "hello"` |
| `str_squish()` | Remove excess whitespace | `str_squish("hello  world") # "hello world"` |
| `str_pad()` | Pad string | `str_pad("abc", 5, "left") # "  abc"` |

## Pattern Matching

### Pattern Types

```r
# Literal strings
str_detect("abc", "a")  # TRUE

# Regular expressions
str_detect("abc", "^a")  # TRUE - starts with "a"

# Fixed strings (faster for exact matching)
str_detect("abc", fixed("a"))  # TRUE

# Collation for locale-sensitive comparison
str_detect("abc", coll("a"))  # TRUE

# Boundary detection
str_detect("abc", boundary("character"))  # TRUE
```

### Regular Expression Functions

| Function | Description | Example |
|----------|-------------|---------|
| `str_extract()` | Extract first match | `str_extract("banana", "na") # "na"` |
| `str_extract_all()` | Extract all matches | `str_extract_all("banana", "na") # list: c("na", "na")` |
| `str_match()` | Extract matched groups | `str_match("abc123", "(\\d+)") # matrix: "123", "123"` |
| `str_match_all()` | Extract all matched groups | `str_match_all("abc123def456", "(\\d+)")` |
| `str_replace()` | Replace first match | `str_replace("banana", "na", "XX") # "baXXna"` |
| `str_replace_all()` | Replace all matches | `str_replace_all("banana", "na", "XX") # "baXXXX"` |
| `str_remove()` | Remove first match | `str_remove("banana", "na") # "bana"` |
| `str_remove_all()` | Remove all matches | `str_remove_all("banana", "na") # "ba"` |
| `str_split()` | Split string | `str_split("a,b,c", ",") # list: c("a", "b", "c")` |
| `str_split_fixed()` | Split to fixed number | `str_split_fixed("a,b,c", ",", 2) # matrix: c("a", "b,c")` |

## Advanced Pattern Matching

### Complex Regular Expressions

```r
# Using regex capture groups
str_match("abc-123", "([a-z]+)-([0-9]+)")  # Returns "abc-123", "abc", "123"

# Named capture groups
str_match("abc-123", "(?<text>[a-z]+)-(?<numbers>[0-9]+)")

# Lookahead and lookbehind
str_extract("abc123", "\\d+(?=\\D)")  # Numbers followed by non-digits
str_extract("abc123", "(?<=\\D)\\d+")  # Numbers preceded by non-digits
```

### Working with Multiple Patterns

```r
# Vector of patterns
patterns <- c("apple", "banana", "orange")
str_detect("I like apples", patterns)  # TRUE, FALSE, FALSE

# Alternation
str_extract_all("apples and bananas", str_c(patterns, collapse="|"))

# Dictionary-based replacements
fruits <- c("apple" = "APPLE", "banana" = "BANANA")
str_replace_all("I eat apple and banana", fixed(names(fruits), ignore_case = TRUE), fruits)
```

## Common Use Cases and Techniques

### Data Cleaning

```r
# Remove special characters 
str_replace_all(text, "[^[:alnum:] ]", "")

# Standardize whitespace
str_squish(text)

# Fix inconsistent capitalization
str_to_title(names)

# Extract specific patterns
phone_numbers <- str_extract_all(text, "\\d{3}[- ]?\\d{3}[- ]?\\d{4}")
```

### Text Parsing

```r
# Extract elements from structured text
str_match(log_entries, "User: (\\w+), Action: (\\w+)")

# Split and recombine
parts <- str_split(addresses, ",")
clean_parts <- lapply(parts, str_trim)
```

### Data Validation

```r
# Check if string matches pattern
is_valid_email <- str_detect(emails, "^[\\w.]+@[\\w.]+\\.\\w+$")

# Validate string length
is_valid_password <- str_length(passwords) >= 8
```

## Best Practices

1. **Use the right pattern type**:
   - `fixed()` for literal strings (faster performance)
   - Default regex patterns for flexible matching
   - `boundary()` for word/sentence/character boundaries
   - `coll()` for locale-sensitive string comparisons

2. **Vectorize operations**:
   - stringr functions work on vectors, avoiding loops
   ```r
   # Good
   clean_names <- str_to_lower(str_trim(names))
   
   # Avoid
   clean_names <- character(length(names))
   for (i in seq_along(names)) {
     clean_names[i] <- str_to_lower(str_trim(names[i]))
   }
   ```

3. **Chain operations with pipes**:
   ```r
   library(magrittr)
   clean_text <- text %>%
     str_to_lower() %>%
     str_replace_all("[^[:alnum:] ]", "") %>%
     str_squish()
   ```

4. **Handle missing values properly**:
   - Most stringr functions return `NA` for `NA` inputs
   - Use `str_replace_na()` to replace NAs with a string

5. **Pre-compile complex patterns**:
   ```r
   # For repeated use of the same pattern
   email_pattern <- regex("^[\\w.]+@[\\w.]+\\.\\w+$")
   is_email <- str_detect(strings, email_pattern)
   ```

6. **Use string interpolation for complex strings**:
   ```r
   # Instead of concatenation
   str_glue("User {user_id} logged in at {timestamp}")
   ```

7. **Be cautious with memory usage**:
   - `str_c(collapse = "")` can be more efficient than repeated concatenation
   - For large text processing, consider using `stringi` directly

8. **Use regex named groups for clarity**:
   ```r
   str_match(dates, "(?<year>\\d{4})-(?<month>\\d{2})-(?<day>\\d{2})")
   ```

## Performance Tips

1. Use `fixed()` when you don't need regex features
2. Avoid repetitive string operations in loops
3. For very large strings, use `stringi` functions directly
4. Pre-allocate result vectors for large operations
5. For simple operations like counting characters, `nchar()` can be faster than `str_length()`

## Common Regular Expression Patterns

| Pattern | Description | Example |
|---------|-------------|---------|
| `\\d` | Digits | `str_extract_all("abc123", "\\d") # c("1", "2", "3")` |
| `\\w` | Word characters | `str_extract_all("a.b_c", "\\w") # c("a", "b", "c")` |
| `\\s` | Whitespace | `str_count("a b c", "\\s") # 2` |
| `[[:alnum:]]` | Alphanumeric chars | `str_extract_all("a1*b2", "[[:alnum:]]") # c("a", "1", "b", "2")` |
| `^` | Start of string | `str_detect("abc", "^a") # TRUE` |
| `$` | End of string | `str_detect("abc", "c$") # TRUE` |
| `\\b` | Word boundary | `str_detect("the theory", "\\bthe\\b") # TRUE` |



# ggplot2 Reference Card

## Table of Contents
- [Basic Structure](#basic-structure)
- [Data Preparation](#data-preparation)
- [Geometries (geoms)](#geometries-geoms)
- [Aesthetics (aes)](#aesthetics-aes)
- [Scales](#scales)
- [Faceting](#faceting)
- [Themes and Appearance](#themes-and-appearance)
- [Statistics (stats)](#statistics-stats)
- [Positioning](#positioning)
- [Coordinate Systems](#coordinate-systems)
- [Annotations and Labels](#annotations-and-labels)
- [Legends](#legends)
- [Multiple Plots](#multiple-plots)
- [Interactive Plots](#interactive-plots)
- [Saving Plots](#saving-plots)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Basic Structure

The basic structure of a ggplot2 plot:

```r
# Load the library
library(ggplot2)

# Basic structure
ggplot(data = <DATA>, mapping = aes(<MAPPINGS>)) +
  <GEOM_FUNCTION>(mapping = aes(<MAPPINGS>), stat = <STAT>, position = <POSITION>) +
  <COORDINATE_FUNCTION> +
  <SCALE_FUNCTION> +
  <THEME_FUNCTION>
```

Example:
```r
# Simple scatterplot
ggplot(data = mtcars, aes(x = wt, y = mpg)) +
  geom_point()
```

## Data Preparation

ggplot2 works best with data in "tidy" format:
- Each variable is a column
- Each observation is a row
- Each value is a cell

```r
# Reshape wide to long format
library(tidyr)
long_data <- pivot_longer(wide_data, 
                         cols = c("col1", "col2"), 
                         names_to = "variable", 
                         values_to = "value")
```

## Geometries (geoms)

### Basic Plots

| Geom Function | Description | Key Aesthetics |
|---------------|-------------|----------------|
| `geom_point()` | Scatterplot | x, y, color, size, shape, alpha |
| `geom_line()` | Line chart | x, y, color, linetype, size, group |
| `geom_bar()` | Bar chart | x, fill, color |
| `geom_histogram()` | Histogram | x, fill, color, binwidth |
| `geom_boxplot()` | Box-and-whisker plot | x, y, fill, color |
| `geom_violin()` | Violin plot | x, y, fill, color |
| `geom_density()` | Density plot | x, fill, color, alpha |
| `geom_smooth()` | Smoothed line | x, y, color, linetype, method |
| `geom_tile()` | Heatmap | x, y, fill |
| `geom_text()` | Text labels | x, y, label |

### Advanced Geoms

| Geom Function | Description | Key Aesthetics |
|---------------|-------------|----------------|
| `geom_jitter()` | Points with random noise | x, y, width, height |
| `geom_errorbar()` | Error bars | x, ymin, ymax |
| `geom_ribbon()` | Ribbon/area between lines | x, ymin, ymax, fill |
| `geom_area()` | Area plot | x, y, fill, group |
| `geom_contour()` | Contour lines | x, y, z |
| `geom_hex()` | Hexagonal binning | x, y |
| `geom_raster()` | Optimized heatmap | x, y, fill |
| `geom_rug()` | Marginal rug plots | x, y |
| `geom_qq()` | Quantile-quantile plot | sample |
| `geom_sf()` | Simple features map | Geometry data |

## Aesthetics (aes)

Map variables to visual properties:

```r
# Basic aesthetics
ggplot(data, aes(x = var1, y = var2, color = var3, size = var4, shape = var5)) +
  geom_point()
```

| Aesthetic | Description | Example |
|-----------|-------------|---------|
| `x`, `y` | Position on x and y axes | `aes(x = wt, y = mpg)` |
| `color` | Color of points or lines | `aes(color = factor(cyl))` |
| `fill` | Fill color of shapes | `aes(fill = factor(cyl))` |
| `size` | Size of points | `aes(size = hp)` |
| `shape` | Shape of points | `aes(shape = factor(am))` |
| `alpha` | Transparency | `aes(alpha = hp)` |
| `linetype` | Type of line | `aes(linetype = factor(cyl))` |
| `group` | Grouping variable | `aes(group = factor(cyl))` |
| `label` | Text label | `aes(label = rownames(mtcars))` |
| `weight` | Weight for statistical calculations | `aes(weight = frequency)` |

## Scales

Customize how aesthetics are mapped to visual properties:

```r
# Color scale example
ggplot(mtcars, aes(x = wt, y = mpg, color = factor(cyl))) +
  geom_point() +
  scale_color_brewer(palette = "Set1")
```

### Common Scale Functions

| Scale Function | Description | Example |
|----------------|-------------|---------|
| `scale_x_continuous()` | Continuous x scale | `scale_x_continuous(limits = c(0, 10), breaks = seq(0, 10, 2))` |
| `scale_y_log10()` | Log 10 y scale | `scale_y_log10()` |
| `scale_fill_brewer()` | ColorBrewer palette for fill | `scale_fill_brewer(palette = "Blues")` |
| `scale_color_manual()` | Custom colors | `scale_color_manual(values = c("red", "blue", "green"))` |
| `scale_size_area()` | Size scale proportional to area | `scale_size_area(max_size = 10)` |
| `scale_shape_manual()` | Custom shapes | `scale_shape_manual(values = c(16, 17, 18))` |
| `scale_alpha_continuous()` | Alpha transparency scale | `scale_alpha_continuous(range = c(0.1, 1))` |
| `scale_x_date()` | Date scale for x axis | `scale_x_date(date_labels = "%b %Y")` |
| `scale_linetype_discrete()` | Line type scale | `scale_linetype_discrete()` |
| `scale_fill_viridis_c()` | Viridis continuous color scale | `scale_fill_viridis_c(option = "magma")` |

### Color Scales

```r
# Discrete color scales
scale_color_brewer(palette = "Set1")  # ColorBrewer palettes
scale_color_viridis_d()               # Viridis discrete
scale_color_manual(values = c("red", "blue", "green"))  # Custom colors

# Continuous color scales
scale_color_gradient(low = "blue", high = "red")  # Two-color gradient
scale_color_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0)  # Diverging
scale_color_viridis_c()               # Viridis continuous
```

## Faceting

Split plot into multiple panels:

```r
# Facet by one variable
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point() +
  facet_wrap(~ cyl)

# Facet by two variables
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point() +
  facet_grid(vs ~ am)
```

| Facet Function | Description | Example |
|----------------|-------------|---------|
| `facet_wrap()` | Wrap facets into grid | `facet_wrap(~ cyl, nrow = 2)` |
| `facet_grid()` | Grid of facets | `facet_grid(vs ~ am)` |
| `facet_null()` | Single panel (default) | `facet_null()` |

### Facet Options

```r
# Common facet_wrap options
facet_wrap(
  ~ variable,
  nrow = 2,               # Number of rows
  ncol = 3,               # Number of columns
  scales = "free",        # Options: "fixed", "free", "free_x", "free_y"
  labeller = label_both,  # Label format (label_both, label_value, etc.)
  strip.position = "top"  # Position of strips: "top", "bottom", "left", "right"
)

# Common facet_grid options
facet_grid(
  rows ~ cols,
  scales = "free",        # Options: "fixed", "free", "free_x", "free_y"
  space = "free",         # Options: "fixed", "free", "free_x", "free_y"
  labeller = label_both,  # Label format
  switch = "y"            # Switch axis labels: "x", "y", "both"
)
```

## Themes and Appearance

Control the overall look of your plot:

```r
# Built-in themes
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point() +
  theme_minimal()
```

### Built-in Themes

| Theme Function | Description |
|----------------|-------------|
| `theme_gray()` | Default theme |
| `theme_bw()` | Black and white |
| `theme_minimal()` | Minimal theme |
| `theme_classic()` | Classic theme |
| `theme_light()` | Light theme |
| `theme_dark()` | Dark theme |
| `theme_void()` | Empty theme |

### Custom Theme Elements

```r
# Custom theme elements
ggplot(mtcars, aes(x = wt, y = mpg, color = factor(cyl))) +
  geom_point() +
  theme(
    # Text elements
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10),
    
    # Background elements
    panel.background = element_rect(fill = "white"),
    panel.grid.major = element_line(color = "grey90", linewidth = 0.5),
    panel.grid.minor = element_line(color = "grey95", linewidth = 0.2),
    plot.background = element_rect(fill = "white"),
    
    # Legend elements
    legend.position = "right",  # "none", "left", "right", "bottom", "top", or c(x, y)
    legend.background = element_rect(fill = "white", color = "grey80"),
    
    # Margins and spacing
    plot.margin = margin(t = 10, r = 10, b = 10, l = 10, unit = "pt"),
    panel.spacing = unit(1, "lines")
  )
```

### Custom Themes

```r
# Create a custom theme
my_theme <- theme_minimal() +
  theme(
    text = element_text(family = "Arial"),
    plot.title = element_text(size = 16, face = "bold"),
    legend.position = "bottom"
  )

# Apply your custom theme
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point() +
  my_theme
```

## Statistics (stats)

Statistical transformations:

```r
# Explicit stats (usually implied by geoms)
ggplot(mtcars, aes(x = wt)) +
  stat_density(geom = "line")

# Equivalent to:
ggplot(mtcars, aes(x = wt)) +
  geom_density()
```

### Common Stats

| Stat Function | Description | Commonly Used With |
|---------------|-------------|-------------------|
| `stat_count()` | Count number of cases | `geom_bar()` |
| `stat_bin()` | Bin data | `geom_histogram()` |
| `stat_density()` | Compute density estimate | `geom_density()` |
| `stat_boxplot()` | Box-and-whisker plot | `geom_boxplot()` |
| `stat_smooth()` | Smoothed conditional mean | `geom_smooth()` |
| `stat_summary()` | Summarize y values at x | `geom_pointrange()` |
| `stat_function()` | Compute function | `geom_line()` |
| `stat_contour()` | Contour lines from 3d data | `geom_contour()` |
| `stat_ecdf()` | Empirical cumulative density function | `geom_step()` |
| `stat_qq()` | Quantile-quantile plot | `geom_point()` |

### Using stat_summary

```r
# Calculate mean and standard error
ggplot(mtcars, aes(x = factor(cyl), y = mpg)) +
  stat_summary(
    fun = mean,
    fun.min = function(x) mean(x) - sd(x) / sqrt(length(x)),
    fun.max = function(x) mean(x) + sd(x) / sqrt(length(x)),
    geom = "pointrange"
  )

# Summarize with multiple functions
ggplot(mtcars, aes(x = factor(cyl), y = mpg)) +
  stat_summary(
    fun.data = "mean_cl_boot",  # bootstrap confidence interval
    geom = "errorbar",
    width = 0.2
  ) +
  stat_summary(
    fun = "mean",
    geom = "point",
    size = 3
  )
```

## Positioning

Control how objects are positioned:

```r
# Stacked bar chart (default)
ggplot(diamonds, aes(x = cut, fill = clarity)) +
  geom_bar()

# Dodge (side-by-side)
ggplot(diamonds, aes(x = cut, fill = clarity)) +
  geom_bar(position = "dodge")

# Fill (100% stacked)
ggplot(diamonds, aes(x = cut, fill = clarity)) +
  geom_bar(position = "fill")

# Jitter (add random noise)
ggplot(mtcars, aes(x = factor(cyl), y = mpg)) +
  geom_point(position = position_jitter(width = 0.2, height = 0))
```

### Position Functions

| Position Function | Description | Example |
|-------------------|-------------|---------|
| `position_stack()` | Stack overlapping objects | `geom_bar(position = "stack")` |
| `position_dodge()` | Dodge overlapping objects | `geom_bar(position = position_dodge(width = 0.9))` |
| `position_fill()` | Stack objects to fill full height | `geom_bar(position = "fill")` |
| `position_jitter()` | Add random noise | `geom_point(position = position_jitter(width = 0.2))` |
| `position_identity()` | Don't adjust position | `geom_point(position = "identity")` |
| `position_nudge()` | Nudge points a fixed distance | `geom_text(position = position_nudge(x = 0.1))` |
| `position_jitterdodge()` | Jitter and dodge | `geom_point(position = position_jitterdodge())` |

## Coordinate Systems

Modify how data coordinates are mapped to the plot:

```r
# Flip x and y axes
ggplot(mtcars, aes(x = factor(cyl), y = mpg)) +
  geom_boxplot() +
  coord_flip()

# Polar coordinates
ggplot(mtcars, aes(x = factor(cyl), fill = factor(vs))) +
  geom_bar() +
  coord_polar(theta = "y")
```

### Coordinate Functions

| Coordinate Function | Description | Example |
|---------------------|-------------|---------|
| `coord_cartesian()` | Cartesian coordinates with zoom | `coord_cartesian(xlim = c(1, 5), ylim = c(10, 30))` |
| `coord_flip()` | Flip x and y axes | `coord_flip()` |
| `coord_fixed()` | Fixed aspect ratio | `coord_fixed(ratio = 1)` |
| `coord_polar()` | Polar coordinates | `coord_polar(theta = "x")` |
| `coord_trans()` | Transformed cartesian | `coord_trans(x = "log10", y = "log10")` |
| `coord_map()` | Map projections | `coord_map("mercator")` |
| `coord_sf()` | Simple features map projections | `coord_sf()` |

## Annotations and Labels

Add text, labels, and annotations to your plot:

```r
# Titles and labels
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point() +
  labs(
    title = "Car Weight vs. Fuel Efficiency",
    subtitle = "Data from the mtcars dataset",
    caption = "Source: Motor Trend, 1974",
    x = "Weight (1000 lbs)",
    y = "Miles per Gallon",
    color = "Cylinders"
  )

# Adding annotations
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point() +
  annotate("text", x = 4, y = 30, label = "High Efficiency") +
  annotate("rect", xmin = 3, xmax = 5, ymin = 20, ymax = 30,
           alpha = 0.2, fill = "blue") +
  annotate("segment", x = 2, xend = 3, y = 15, yend = 20,
           arrow = arrow(), color = "red")
```

### Text Labels

```r
# Add text labels directly to points
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point() +
  geom_text(aes(label = rownames(mtcars)), nudge_x = 0.1)

# Avoid overlapping with ggrepel
library(ggrepel)
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point() +
  geom_text_repel(aes(label = rownames(mtcars)))
```

### Other Annotations

```r
# Add a reference line
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point() +
  geom_hline(yintercept = 20, linetype = "dashed", color = "red") +
  geom_vline(xintercept = 3, linetype = "dotted", color = "blue")

# Add a shaded rectangle
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point() +
  annotate("rect", xmin = 3, xmax = 4, ymin = 15, ymax = 25,
           alpha = 0.2, fill = "yellow")
```

## Legends

Customize and control the appearance of legends:

```r
# Control legend position and appearance
ggplot(mtcars, aes(x = wt, y = mpg, color = factor(cyl), shape = factor(am))) +
  geom_point(size = 3) +
  scale_color_brewer(palette = "Set1") +
  labs(color = "Cylinders", shape = "Transmission") +
  theme(
    legend.position = "bottom",
    legend.box = "horizontal",
    legend.background = element_rect(fill = "lightgrey"),
    legend.key = element_rect(fill = "white"),
    legend.title = element_text(face = "bold")
  )
```

### Legend Options

```r
# Remove legends
ggplot(mtcars, aes(x = wt, y = mpg, color = factor(cyl))) +
  geom_point() +
  guides(color = "none")

# Customize legend order and appearance
ggplot(mtcars, aes(x = wt, y = mpg, color = factor(cyl), shape = factor(am))) +
  geom_point(size = 3) +
  guides(
    color = guide_legend(
      title = "Number of Cylinders",
      override.aes = list(size = 5),
      nrow = 1,
      order = 1
    ),
    shape = guide_legend(
      title = "Transmission",
      override.aes = list(size = 3),
      order = 2
    )
  )
```

## Multiple Plots

Combine multiple plots into a single figure:

```r
# Using patchwork package (recommended)
library(patchwork)

p1 <- ggplot(mtcars, aes(x = wt, y = mpg)) + geom_point()
p2 <- ggplot(mtcars, aes(x = hp, y = mpg)) + geom_point()
p3 <- ggplot(mtcars, aes(x = mpg)) + geom_histogram()

# Arrange plots side by side
p1 + p2

# Arrange plots in a grid
(p1 + p2) / p3

# Add a title to the combined plot
(p1 + p2) / p3 + 
  plot_annotation(
    title = "Car Efficiency Relationships",
    subtitle = "Exploring the mtcars dataset",
    tag_levels = "A"
  )
```

### Using gridExtra

```r
# Using gridExtra package
library(gridExtra)

p1 <- ggplot(mtcars, aes(x = wt, y = mpg)) + geom_point()
p2 <- ggplot(mtcars, aes(x = hp, y = mpg)) + geom_point()

grid.arrange(p1, p2, ncol = 2)
```

## Interactive Plots

Turn static plots into interactive ones:

```r
# Using plotly
library(plotly)

p <- ggplot(mtcars, aes(x = wt, y = mpg, color = factor(cyl))) +
  geom_point() +
  labs(title = "Weight vs MPG")

ggplotly(p)

# Using ggiraph
library(ggiraph)

p <- ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point_interactive(aes(tooltip = rownames(mtcars), data_id = rownames(mtcars))) +
  labs(title = "Weight vs MPG")

girafe(ggobj = p)
```

## Saving Plots

Save your plots to various file formats:

```r
# Basic usage
p <- ggplot(mtcars, aes(x = wt, y = mpg)) + geom_point()

ggsave("myplot.pdf", plot = p, width = 8, height = 6, units = "in")
```

### Common ggsave Options

```r
ggsave(
  filename = "myplot.png",  # File name with extension
  plot = p,                 # Plot object (optional, uses last plot by default)
  device = "png",           # File format (optional, inferred from filename)
  path = "figures/",        # Path to save to
  width = 8,                # Width
  height = 6,               # Height
  units = "in",             # Units: "in", "cm", "mm", or "px"
  dpi = 300,                # Resolution in dots per inch
  bg = "white",             # Background color
  scale = 1                 # Scaling factor
)
```

## Best Practices

### Data Preparation
1. Use tidy data (one variable per column, one observation per row)
2. Convert categorical variables to factors with meaningful levels
3. Handle missing values before plotting

### Plot Design
1. Start with a clear purpose - what specific question are you trying to answer?
2. Choose the appropriate visualization for your data type and question
3. Focus on the data, minimize non-data ink
4. Use color meaningfully, not just for decoration
5. Ensure text is readable (labels, titles, annotations)
6. Consider your audience when determining the level of complexity

### Code Structure
1. Build plots in layers, one step at a time
2. Save intermediate objects for complex plots
3. Create custom themes and functions for consistent plots
4. Use comments to document your plotting decisions

### Color Usage
1. Use colorblind-friendly palettes (viridis, ColorBrewer)
2. Limit the number of distinct colors (5-7 maximum)
3. Use sequential or diverging palettes for continuous data
4. Consider cultural meanings of colors

```r
# Use viridis for colorblind-friendly continuous scales
ggplot(mtcars, aes(x = wt, y = mpg, color = disp)) +
  geom_point(size = 3) +
  scale_color_viridis_c() +
  theme_minimal()

# Use ColorBrewer for categorical data
ggplot(mtcars, aes(x = wt, y = mpg, color = factor(cyl))) +
  geom_point(size = 3) +
  scale_color_brewer(palette = "Set2") +
  theme_minimal()
```

### Performance
1. For large datasets, consider using `geom_hex()` or `geom_bin2d()` instead of `geom_point()`
2. Set `alpha` for transparency with overlapping points
3. Use `ggplot2::labs()` instead of individual functions for titles and labels

## Troubleshooting

### Common Errors

| Error Message | Possible Cause | Solution |
|---------------|----------------|----------|
| "Aesthetics must be either length 1 or the same as the data" | Mismatched data dimensions | Check length of vectors in aes() |
| "Object not found" | Missing variable | Check variable names, use `dplyr::select()` |
| "Don't know how to automatically pick scale for object" | Incorrect aesthetic mapping | Verify data types match aesthetics |
| "Cannot add ggproto objects together" | Missing + operator | Check for missing + between layers |
| "Position guide requires single variable" | Using two variables for one position | Map only one variable to x or y |

### Debugging Strategies

1. Build your plot incrementally, adding one layer at a time
2. Check your data structure with `str()` and `head()`
3. Verify factor levels for categorical variables
4. Use `aes(text = ...)` with `ggplotly()` to inspect values
5. For complex plots, save intermediate objects to inspect each step

```r
# Debugging example
# Examine data structure
str(mtcars)

# Start with a basic plot and add one layer at a time
p1 <- ggplot(mtcars, aes(x = wt, y = mpg))
p1  # View the empty plot

p2 <- p1 + geom_point()
p2  # Check points are added correctly

p3 <- p2 + facet_wrap(~ cyl)
p3  # Verify faceting

# If something goes wrong, you can identify which step caused the issue
```



# R Shiny Reference Card

## Basics

### Core Structure
```r
library(shiny)

ui <- fluidPage(
  # UI elements go here
)

server <- function(input, output, session) {
  # Server logic goes here
}

shinyApp(ui = ui, server = server)
```

### UI Layout Options

#### Page Layouts
- `fluidPage()` - Responsive layout that adjusts to browser size
- `fixedPage()` - Fixed-width layout
- `navbarPage()` - Page with navigation bar
- `dashboardPage()` - Dashboard layout (requires `shinydashboard` package)
- `bootstrapPage()` - Minimalist page with Bootstrap

#### Layout Functions
- `fluidRow()` and `column()` - Grid system
- `wellPanel()` - Gray container with inset appearance
- `tabsetPanel()` and `tabPanel()` - Tabbed interface
- `navlistPanel()` - Sidebar navigation list
- `sidebarLayout()`, `sidebarPanel()`, `mainPanel()` - Side-by-side panels
- `splitLayout()` - Split screen horizontally
- `verticalLayout()` - Stack elements vertically

## Input Elements

### Text Inputs
```r
# Basic text input
textInput("id", "Label", "default value")

# Numeric input
numericInput("id", "Label", value = 0, min = 0, max = 100, step = 1)

# Password input
passwordInput("id", "Label")

# Text area (multi-line)
textAreaInput("id", "Label", rows = 3)
```

### Selection Inputs
```r
# Dropdown selection
selectInput("id", "Label", choices = c("A", "B", "C"), selected = "A", multiple = FALSE)

# Radio buttons
radioButtons("id", "Label", choices = c("A", "B", "C"), selected = "A")

# Checkbox group
checkboxGroupInput("id", "Label", choices = c("A", "B", "C"), selected = "A")

# Single checkbox
checkboxInput("id", "Label", value = FALSE)
```

### Numeric Inputs
```r
# Slider
sliderInput("id", "Label", min = 0, max = 100, value = 50)

# Range slider
sliderInput("id", "Label", min = 0, max = 100, value = c(25, 75))

# Date input
dateInput("id", "Label", value = Sys.Date())

# Date range
dateRangeInput("id", "Label", start = Sys.Date() - 7, end = Sys.Date())
```

### File Inputs
```r
fileInput("id", "Label", multiple = TRUE, accept = c(".csv", ".txt"))
```

### Action Elements
```r
# Button
actionButton("id", "Label")

# Link that acts like a button
actionLink("id", "Link text")

# Download button
downloadButton("id", "Download")
downloadLink("id", "Download link")
```

## Output Elements

### Basic Outputs
```r
# Text output
textOutput("id")  # For regular text
verbatimTextOutput("id")  # For code/console-like output

# HTML output
htmlOutput("id")
uiOutput("id")  # Same as htmlOutput, for dynamic UI elements

# Image output 
imageOutput("id")

# Table output
tableOutput("id")  # For small tables
dataTableOutput("id")  # For large, interactive tables (DT package)
```

### Plot Outputs
```r
# Plot output
plotOutput("id", 
           hover = hoverOpts("plot_hover"),  # For hover interactivity
           click = clickOpts("plot_click"),  # For click interactivity
           brush = brushOpts("plot_brush")   # For area selection
)
```

## Server Logic

### Reactive Programming

#### Rendering Outputs
```r
# Text
output$text1 <- renderText({
  paste("The value is", input$slider)
})

# Tables
output$table1 <- renderTable({
  head(mtcars)
})
output$table2 <- renderDataTable({
  mtcars
})

# Plots
output$plot1 <- renderPlot({
  hist(rnorm(input$slider))
})

# Images
output$image1 <- renderImage({
  list(src = "path/to/image.png", contentType = "image/png", width = 400, height = 300)
}, deleteFile = FALSE)

# UI elements
output$dynamic_ui <- renderUI({
  selectInput("dynamic", "Dynamic choices", choices = 1:input$slider)
})
```

#### Reactive Expressions
```r
# Create a reactive expression
filtered_data <- reactive({
  subset(mtcars, cyl == input$cylinders)
})

# Use the reactive expression in outputs
output$plot <- renderPlot({
  plot(filtered_data()$mpg, filtered_data()$hp)
})
```

#### Reactive Values
```r
# Create a reactiveValues object
values <- reactiveValues(counter = 0)

# Update and use reactive values
observeEvent(input$increment, {
  values$counter <- values$counter + 1
})

output$result <- renderText({
  paste("Counter:", values$counter)
})
```

#### Event Handling
```r
# Run code when a button is clicked
observeEvent(input$button, {
  # Code to execute when button is clicked
})

# Run code when any reactive value changes
observe({
  # Code that runs whenever reactive dependencies change
})
```

#### Isolation
```r
# Prevent reactivity
isolate({
  # Code here won't create reactive dependencies
  current_value <- input$slider
})
```

## Advanced Concepts

### Modules
```r
# Module UI function
counterUI <- function(id) {
  ns <- NS(id)  # Create namespaced IDs
  tagList(
    actionButton(ns("increment"), "Increment"),
    textOutput(ns("value"))
  )
}

# Module server function
counterServer <- function(id) {
  moduleServer(id, function(input, output, session) {
    count <- reactiveVal(0)
    
    observeEvent(input$increment, {
      count(count() + 1)
    })
    
    output$value <- renderText({
      count()
    })
    
    return(count)  # Can return reactive expressions
  })
}

# Using modules in app
ui <- fluidPage(
  counterUI("counter1"),
  counterUI("counter2")
)

server <- function(input, output, session) {
  count1 <- counterServer("counter1")
  count2 <- counterServer("counter2")
}
```

### Bookmarking
```r
# Enable bookmarking
enableBookmarking(store = "url")  # or "server"

# In UI
ui <- function(request) {
  fluidPage(
    bookmarkButton(),
    # UI elements
  )
}

# Custom bookmarking logic
onBookmark(function(state) {
  # Add custom values to state$values
})

onRestore(function(state) {
  # Handle custom values from state$values
})
```

### Dynamic UI
```r
# Insert UI elements dynamically
insertUI(
  selector = "#placeholder",
  where = "afterEnd",
  ui = textInput("dynamic", "Dynamic input")
)

# Remove UI elements
removeUI(
  selector = "#dynamic-input"
)
```

### Async Operations
```r
library(future)
library(promises)

# Set up future plan
plan(multisession)

# Create a promise
promise <- future({ 
  # Long running computation
  Sys.sleep(5)
  return(42)
}) %...>%
  (function(value) {
    # Handle result
    return(value * 2)
  }) %...!%
  (function(error) {
    # Handle error
    return(NA)
  })

# Use promise in a render function
output$result <- renderText({
  promise
})
```

## Data Handling

### File Upload and Download
```r
# Server-side upload handling
output$fileInfo <- renderTable({
  req(input$file)
  input$file
})

# Reading uploaded file
data <- reactive({
  req(input$file)
  read.csv(input$file$datapath)
})

# File download
output$downloadData <- downloadHandler(
  filename = function() {
    paste("data-", Sys.Date(), ".csv", sep = "")
  },
  content = function(file) {
    write.csv(filtered_data(), file, row.names = FALSE)
  }
)
```

### Database Connections
```r
library(DBI)
library(RSQLite)

# Connect to database once
conn <- NULL
onStart <- function() {
  conn <<- dbConnect(RSQLite::SQLite(), "my_database.sqlite")
}

onStop <- function() {
  if (!is.null(conn)) {
    dbDisconnect(conn)
  }
}

# Query data
data <- reactive({
  query <- paste0("SELECT * FROM mytable WHERE category = '", input$category, "'")
  dbGetQuery(conn, query)
})
```

## Extensions & Integration

### htmlwidgets Integration
```r
library(plotly)
library(DT)

# Plotly
output$plotly <- renderPlotly({
  plot_ly(mtcars, x = ~wt, y = ~mpg, color = ~factor(cyl))
})

# DataTables
output$dt <- renderDT({
  datatable(mtcars, 
            options = list(pageLength = 10),
            filter = 'top'
  )
})
```

### Shiny Gadgets
```r
selectPointsGadget <- function(data, xvar, yvar) {
  ui <- miniPage(
    gadgetTitleBar("Select Points"),
    miniContentPanel(
      plotOutput("plot", brush = brushOpts("brush", clip = TRUE))
    )
  )
  
  server <- function(input, output, session) {
    output$plot <- renderPlot({
      plot(data[[xvar]], data[[yvar]], xlab = xvar, ylab = yvar)
    })
    
    observeEvent(input$done, {
      brushed_points <- brushedPoints(data, input$brush, xvar, yvar)
      stopApp(brushed_points)
    })
  }
  
  runGadget(ui, server)
}
```

### JavaScript Integration
```r
# Custom input binding
tags$script("
  $(function() {
    var customInputBinding = new Shiny.InputBinding();
    $.extend(customInputBinding, {
      find: function(scope) {
        return $(scope).find('.custom-input');
      },
      getValue: function(el) {
        return $(el).data('value');
      },
      setValue: function(el, value) {
        $(el).data('value', value);
      },
      subscribe: function(el, callback) {
        $(el).on('change.customInputBinding', function() {
          callback();
        });
      },
      unsubscribe: function(el) {
        $(el).off('.customInputBinding');
      }
    });

    Shiny.inputBindings.register(customInputBinding);
  });
")
```

## Best Practices

### Performance Optimization
1. **Use reactive expressions** to cache calculations
   ```r
   # Inefficient: Recalculates for each output
   output$plot1 <- renderPlot({ calculate_data(input$x) })
   output$plot2 <- renderPlot({ calculate_data(input$x) })
   
   # Efficient: Calculate once, use many times
   calculated_data <- reactive({ calculate_data(input$x) })
   output$plot1 <- renderPlot({ use_data(calculated_data()) })
   output$plot2 <- renderPlot({ use_data(calculated_data()) })
   ```

2. **Debounce inputs** to reduce calculation frequency
   ```r
   sliderInput("slider", "Value:", min = 1, max = 100, value = 50, 
              animate = animationOptions(interval = 300))
   ```

3. **Use bindCache** for expensive rendering operations
   ```r
   output$plot <- renderPlot({
     # Expensive plot
     plot(huge_calculation(input$x))
   }) %>% bindCache(input$x)
   ```

4. **Lazy loading of UI elements** using `tabsetPanel` or conditional panels

5. **Use dataTableOutput** instead of tableOutput for large tables

### Code Organization
1. **Modularize code** with Shiny modules
2. **Separate UI and server** code into different files
3. **Create utility functions** for repeated operations
4. **Use global.R** for loading packages and data preprocessing
5. **Store constants** at the top of your script

### Error Handling
1. **Use req() to validate inputs**
   ```r
   output$plot <- renderPlot({
     req(input$dataset, input$variable)
     # Plot code here
   })
   ```

2. **Add validate() for user-friendly error messages**
   ```r
   output$plot <- renderPlot({
     validate(
       need(input$x != "", "Please select an X variable"),
       need(input$y != "", "Please select a Y variable")
     )
     # Plot code here
   })
   ```

3. **Try-catch for error handling**
   ```r
   output$result <- renderText({
     tryCatch({
       # Code that might fail
     }, error = function(e) {
       return(paste("An error occurred:", e$message))
     })
   })
   ```

### Testing
1. **Use shinytest2 package** for automated testing
2. **Create unit tests** for non-Shiny calculation functions
3. **Test reactivity** with testServer function
   ```r
   library(shinytest2)
   
   test_that("counter increments correctly", {
     testServer(counterServer, {
       session$setInputs(increment = 1)
       expect_equal(count(), 1)
       
       session$setInputs(increment = 1)
       expect_equal(count(), 2)
     })
   })
   ```

### Deployment
1. **Use shinyapps.io** for easy cloud deployment
2. **Consider Shiny Server** for on-premises hosting
3. **Dockerize applications** for consistent environments
4. **Set resource limits** to prevent server overload
5. **Monitor application** with built-in or custom logs

## Example Applications

### Interactive Data Explorer
```r
library(shiny)
library(ggplot2)
library(dplyr)

ui <- fluidPage(
  titlePanel("Dataset Explorer"),
  sidebarLayout(
    sidebarPanel(
      selectInput("dataset", "Choose dataset:", 
                  choices = c("mtcars", "iris", "diamonds")),
      uiOutput("x_var"),
      uiOutput("y_var"),
      checkboxInput("smooth", "Add smoothing line", FALSE)
    ),
    mainPanel(
      plotOutput("scatter"),
      dataTableOutput("data_table")
    )
  )
)

server <- function(input, output, session) {
  # Get the selected dataset
  data <- reactive({
    switch(input$dataset,
           "mtcars" = mtcars,
           "iris" = iris,
           "diamonds" = diamonds %>% sample_n(1000))
  })
  
  # Dynamically generate UI for variable selection
  output$x_var <- renderUI({
    selectInput("x", "X variable:", names(data()))
  })
  
  output$y_var <- renderUI({
    selectInput("y", "Y variable:", names(data()), 
                selected = names(data())[2])
  })
  
  # Create scatter plot
  output$scatter <- renderPlot({
    req(input$x, input$y)
    
    p <- ggplot(data(), aes_string(x = input$x, y = input$y)) +
      geom_point(alpha = 0.7) +
      theme_minimal()
    
    if(input$smooth) {
      p <- p + geom_smooth(method = "loess")
    }
    
    p
  })
  
  # Show data table
  output$data_table <- renderDataTable({
    data()
  }, options = list(pageLength = 10))
}

shinyApp(ui, server)
```

### Reactive Dashboard
```r
library(shiny)
library(shinydashboard)
library(ggplot2)
library(dplyr)

ui <- dashboardPage(
  dashboardHeader(title = "Sales Dashboard"),
  dashboardSidebar(
    dateRangeInput("daterange", "Date range:",
                   start = "2023-01-01", end = "2023-12-31"),
    selectInput("region", "Region:",
                choices = c("All", "North", "South", "East", "West"),
                selected = "All"),
    checkboxGroupInput("product", "Product:",
                       choices = c("A", "B", "C", "D"),
                       selected = c("A", "B", "C", "D"))
  ),
  dashboardBody(
    fluidRow(
      valueBoxOutput("total_sales", width = 4),
      valueBoxOutput("avg_order", width = 4),
      valueBoxOutput("num_orders", width = 4)
    ),
    fluidRow(
      box(plotOutput("sales_trend"), title = "Sales Trend", width = 8),
      box(plotOutput("product_split"), title = "Product Split", width = 4)
    ),
    fluidRow(
      box(DT::dataTableOutput("sales_table"), 
          title = "Top Orders", width = 12)
    )
  )
)

server <- function(input, output, session) {
  # Simulate sales data
  sales_data <- reactive({
    set.seed(123)
    dates <- seq(as.Date("2023-01-01"), as.Date("2023-12-31"), by = "day")
    regions <- c("North", "South", "East", "West")
    products <- c("A", "B", "C", "D")
    
    n <- 1000
    data.frame(
      date = sample(dates, n, replace = TRUE),
      region = sample(regions, n, replace = TRUE),
      product = sample(products, n, replace = TRUE),
      amount = runif(n, 100, 1000)
    )
  })
  
  # Filter data based on inputs
  filtered_data <- reactive({
    data <- sales_data()
    
    # Filter by date
    data <- data %>%
      filter(date >= input$daterange[1] & date <= input$daterange[2])
    
    # Filter by region
    if (input$region != "All") {
      data <- data %>% filter(region == input$region)
    }
    
    # Filter by product
    data <- data %>% filter(product %in% input$product)
    
    data
  })
  
  # Value boxes
  output$total_sales <- renderValueBox({
    total <- sum(filtered_data()$amount)
    valueBox(
      paste0("$", format(round(total), big.mark = ",")),
      "Total Sales",
      icon = icon("dollar-sign"),
      color = "green"
    )
  })
  
  output$avg_order <- renderValueBox({
    avg <- mean(filtered_data()$amount)
    valueBox(
      paste0("$", round(avg)),
      "Average Order",
      icon = icon("shopping-cart"),
      color = "blue"
    )
  })
  
  output$num_orders <- renderValueBox({
    n <- nrow(filtered_data())
    valueBox(
      n,
      "Number of Orders",
      icon = icon("list"),
      color = "purple"
    )
  })
  
  # Sales trend plot
  output$sales_trend <- renderPlot({
    df <- filtered_data() %>%
      group_by(date = as.Date(date)) %>%
      summarize(sales = sum(amount))
    
    ggplot(df, aes(x = date, y = sales)) +
      geom_line() +
      geom_smooth(method = "loess", se = FALSE, color = "red") +
      theme_minimal() +
      labs(x = "Date", y = "Sales ($)")
  })
  
  # Product split plot
  output$product_split <- renderPlot({
    df <- filtered_data() %>%
      group_by(product) %>%
      summarize(sales = sum(amount))
    
    ggplot(df, aes(x = "", y = sales, fill = product)) +
      geom_bar(stat = "identity", width = 1) +
      coord_polar("y", start = 0) +
      theme_minimal() +
      labs(fill = "Product")
  })
  
  # Data table
  output$sales_table <- DT::renderDataTable({
    filtered_data() %>%
      arrange(desc(amount)) %>%
      head(100) %>%
      mutate(amount = paste0("$", round(amount, 2)))
  })
}

shinyApp(ui, server)
```


