#'
#' This function creates a gif from a list of plots.
#'
#' @param plots a list of plots
#' @param filename the name of the gif file to create
#' @param delay
#' @param base_height the height of the plots and resulting gif file, default to 15 inches
#' @param base_width the height of the plots and resulting gif file, default to 1.61 * base_height
#'
createGif <- function(plots, filename, delay = 80, base_height = 3.71, base_width = NULL) {
  # Create the directory where the png will be saved (root of filename)
  temp_dir <- stringi::stri_replace_first_regex(filename, "(?<=/).[^/]*$", "")
  temp_dir <- stringi::stri_replace_first_regex(temp_dir, "/$", "")
  temp_dir <- glue::glue("{temp_dir}_temp/")
  dir.create(temp_dir)
  # Define the base width
  base_width = base_height * 1.61
  # Save the plots as png in the temp dir
  for (i in seq_len(length(plots))) {
    cowplot::save_plot(
      filename = glue::glue("{temp_dir}plot_{stringi::stri_pad_left(i, nchar(length(plots)) + 1, '0')}.jpeg"),
      plot = plots[[i]],
      base_height = base_height,
      base_width = base_width
    )
  }
  # Build the gif
  system(glue::glue("convert -delay {delay} {temp_dir}*.jpeg {filename}"))
  # Remove the temp dir
  unlink(temp_dir, recursive = TRUE)
}
