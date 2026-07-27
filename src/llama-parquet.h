#pragma once

#include <string>

std::string llama_parquet_load_text(const char * path, const char * requested_column);
