parse_bool <- function(x) {
  tolower(trimws(as.character(x))) %in% c("true", "t", "1", "yes", "y")
}

symptom_order <- c(
  "V0注意缺陷", "V0多动冲动", "0总分", "品行", "学习", "心身", "冲动多动", "焦虑",
  "多动指数", "EL", "M抑制", "M转换", "M情感控制", "M启动", "M工作记忆", "M计划",
  "M组织", "M监控", "M行为管理指数", "M元认知指数", "M总分"
)

modality_order <- c("eeg", "gmv", "fcs")

make_color_map <- function(max_abs) {
  # Match the original red-blue style used in significant_fdr_heatmaps.
  palette_values <- grDevices::colorRampPalette(c("#2166AC", "#F7F7F7", "#B2182B"))(201)
  function(value) {
    if (is.na(value)) {
      return("#FFFFFF")
    }
    if (!is.finite(max_abs) || max_abs <= 0) {
      return("#F7F7F7")
    }
    clipped <- max(min(value, max_abs), -max_abs)
    index <- round((clipped + max_abs) / (2 * max_abs) * 200) + 1
    palette_values[index]
  }
}

draw_empty_plot <- function(output_file, title_text, symptom_count) {
  grDevices::cairo_pdf(output_file, width = 7, height = 11, family = "Microsoft YaHei")
  op <- par(no.readonly = TRUE)
  on.exit({
    par(op)
    grDevices::dev.off()
  }, add = TRUE)
  par(mar = c(2, 2, 4, 2))
  plot.new()
  title(main = title_text, cex.main = 1.1)
  text(0.5, 0.56, "No rows with significant_fdr == TRUE", cex = 1.3)
  text(0.5, 0.48, paste("Symptoms in file:", symptom_count), cex = 1.0, col = "#555555")
}

draw_heatmap <- function(mat, sig_mat, output_file, title_text) {
  n_rows <- nrow(mat)
  n_cols <- ncol(mat)
  max_abs <- max(abs(mat), na.rm = TRUE)
  if (!is.finite(max_abs) || max_abs <= 0) {
    max_abs <- 1
  }

  color_for <- make_color_map(max_abs)
  width_px <- max(2000, 260 * n_cols + 1000)
  height_px <- max(3000, 140 * n_rows + 800)
  if (height_px <= width_px) {
    height_px <- width_px + 600
  }

  pdf_width <- max(8.5, width_px / 260)
  pdf_height <- max(11, height_px / 260)
  grDevices::cairo_pdf(output_file, width = pdf_width, height = pdf_height, family = "Microsoft YaHei")
  op <- par(no.readonly = TRUE)
  on.exit({
    par(op)
    grDevices::dev.off()
  }, add = TRUE)

  layout(matrix(c(1, 2), nrow = 1), widths = c(9, 1.6))

  plot_mat <- mat[rev(seq_len(n_rows)), , drop = FALSE]
  row_labels <- rownames(plot_mat)
  col_labels <- colnames(plot_mat)

  par(mar = c(6, 8, 3, 1))
  plot(
    NA,
    xlim = c(0, n_cols),
    ylim = c(0, n_rows),
    xaxs = "i",
    yaxs = "i",
    axes = FALSE,
    xlab = "",
    ylab = "",
    main = title_text
  )

  for (row_idx in seq_len(n_rows)) {
    for (col_idx in seq_len(n_cols)) {
      value <- plot_mat[row_idx, col_idx]
      rect(
        xleft = col_idx - 1,
        ybottom = row_idx - 1,
        xright = col_idx,
        ytop = row_idx,
        col = color_for(value),
        border = "#D9D9D9",
        lwd = 0.8
      )
      if (isTRUE(sig_mat[rev(seq_len(n_rows))[row_idx], col_idx])) {
        text(x = col_idx - 0.5, y = row_idx - 0.5, labels = "*", cex = 1.2, font = 2, col = "#111111")
      }
    }
  }

  axis(side = 1, at = seq_len(n_cols) - 0.5, labels = col_labels, las = 2, tick = FALSE, cex.axis = if (n_cols > 12) 0.75 else 0.9)
  axis(side = 2, at = seq_len(n_rows) - 0.5, labels = row_labels, las = 2, tick = FALSE, cex.axis = 0.95)
  box(col = "#333333", lwd = 1)
  mtext("Feature (modality + feature_index)", side = 1, line = 4.5)
  mtext("Symptom", side = 2, line = 6)

  par(mar = c(6, 1, 3, 4))
  plot(NA, xlim = c(0, 1), ylim = c(-max_abs, max_abs), xaxs = "i", yaxs = "i", axes = FALSE, xlab = "", ylab = "")
  legend_breaks <- seq(-max_abs, max_abs, length.out = 200)
  for (idx in seq_len(length(legend_breaks) - 1)) {
    rect(0, legend_breaks[idx], 1, legend_breaks[idx + 1], col = color_for(mean(legend_breaks[idx:(idx + 1)])), border = NA)
  }
  box(col = "#333333", lwd = 1)
  axis(side = 4, at = pretty(c(-max_abs, max_abs), n = 5), las = 2, cex.axis = 0.9)
  mtext("partial_r", side = 4, line = 3)
}

build_matrix <- function(df, all_symptoms) {
  df$partial_r <- suppressWarnings(as.numeric(df$partial_r))
  df$feature_index_num <- suppressWarnings(as.numeric(df$feature_index))
  df$is_significant <- parse_bool(df$significant_fdr)
  df$feature_label <- paste0(tolower(df$modality), "\n", df$feature_index)
  df$modality_rank <- match(tolower(df$modality), modality_order)
  df$modality_rank[is.na(df$modality_rank)] <- length(modality_order) + 1

  sig_features <- unique(df$feature_label[df$is_significant])
  feature_order_df <- unique(df[, c("modality", "feature_index", "feature_index_num", "feature_label", "modality_rank")])
  feature_order_df <- feature_order_df[feature_order_df$feature_label %in% sig_features, , drop = FALSE]
  feature_order_df <- feature_order_df[
    order(feature_order_df$modality_rank, feature_order_df$feature_index_num, feature_order_df$feature_index),
    ,
    drop = FALSE
  ]
  feature_levels <- unique(feature_order_df$feature_label)

  mat <- matrix(NA_real_, nrow = length(all_symptoms), ncol = length(feature_levels), dimnames = list(all_symptoms, feature_levels))
  sig_mat <- matrix(FALSE, nrow = length(all_symptoms), ncol = length(feature_levels), dimnames = list(all_symptoms, feature_levels))

  df_use <- df[df$feature_label %in% feature_levels, , drop = FALSE]
  for (idx in seq_len(nrow(df_use))) {
    row_name <- df_use$symptom[idx]
    col_name <- df_use$feature_label[idx]
    value <- df_use$partial_r[idx]
    is_sig <- isTRUE(df_use$is_significant[idx])
    if (is.na(value) || is.na(row_name) || is.na(col_name)) {
      next
    }
    current_value <- mat[row_name, col_name]
    current_sig <- sig_mat[row_name, col_name]
    if (is.na(current_value) || (is_sig && !current_sig) || (is_sig == current_sig && abs(value) > abs(current_value))) {
      mat[row_name, col_name] <- value
      sig_mat[row_name, col_name] <- is_sig
    } else if (is_sig) {
      sig_mat[row_name, col_name] <- TRUE
    }
  }

  list(mat = mat, sig_mat = sig_mat)
}

read_results_csv <- function(csv_file) {
  con <- file(csv_file, open = "rt", encoding = "UTF-8-BOM")
  on.exit(close(con), add = TRUE)
  read.csv(con, stringsAsFactors = FALSE, check.names = FALSE)
}

plot_one_file <- function(csv_file, output_file, title_text) {
  df <- read_results_csv(csv_file)
  required_columns <- c("modality", "feature_index", "symptom", "partial_r", "significant_fdr")
  missing_columns <- setdiff(required_columns, names(df))
  if (length(missing_columns) > 0) {
    stop(sprintf("Missing required columns in %s: %s", basename(csv_file), paste(missing_columns, collapse = ", ")))
  }

  symptoms_in_file <- unique(df$symptom)
  ordered_symptoms <- symptom_order[symptom_order %in% symptoms_in_file]
  remaining_symptoms <- setdiff(symptoms_in_file, symptom_order)
  all_symptoms <- c(ordered_symptoms, remaining_symptoms)
  sig_count <- sum(parse_bool(df$significant_fdr), na.rm = TRUE)

  if (sig_count == 0) {
    draw_empty_plot(output_file, title_text, length(all_symptoms))
    return(data.frame(csv_file = csv_file, output_file = output_file, significant_hits = 0))
  }

  built <- build_matrix(df, all_symptoms)
  draw_heatmap(built$mat, built$sig_mat, output_file, title_text)
  data.frame(csv_file = csv_file, output_file = output_file, significant_hits = sig_count)
}

main <- function() {
  root_dir <- getwd()
  input_files <- data.frame(
    dataset = c("HX", "Tt"),
    csv_file = c(
      file.path(root_dir, "analysis_outputs_followup_med1_partialcorr", "HX", "all_results.csv"),
      file.path(root_dir, "analysis_outputs_followup_med1_partialcorr", "Tt", "all_results.csv")
    ),
    output_file = c(
      file.path(root_dir, "analysis_outputs_followup_med1_partialcorr", "HX", "HX_significant_fdr_heatmap.pdf"),
      file.path(root_dir, "analysis_outputs_followup_med1_partialcorr", "Tt", "Tt_significant_fdr_heatmap.pdf")
    ),
    stringsAsFactors = FALSE
  )

  summaries <- lapply(
    seq_len(nrow(input_files)),
    function(idx) {
      plot_one_file(input_files$csv_file[idx], input_files$output_file[idx], paste0(input_files$dataset[idx], " significant_fdr heatmap"))
    }
  )

  summary_df <- do.call(rbind, summaries)
  write.csv(summary_df, file.path(root_dir, "analysis_outputs_followup_med1_partialcorr", "heatmap_summary.csv"), row.names = FALSE, fileEncoding = "UTF-8")
}

if (sys.nframe() == 0) {
  main()
}
