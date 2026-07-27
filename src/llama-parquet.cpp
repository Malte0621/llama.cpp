#include "llama-parquet.h"

#include "llama-impl.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <memory>
#include <stdexcept>
#include <string>

#ifdef LLAMA_USE_PARQUET
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

static bool is_parquet_text_type(arrow::Type::type type) {
    return type == arrow::Type::STRING || type == arrow::Type::LARGE_STRING ||
            type == arrow::Type::BINARY || type == arrow::Type::LARGE_BINARY;
}

template<typename Array>
static void append_parquet_strings(const Array & array, std::string & text) {
    for (int64_t i = 0; i < array.length(); ++i) {
        if (array.IsNull(i)) {
            continue;
        }
        const auto value = array.GetView(i);
        const size_t length = size_t(value.size());
        if (length >= size_t(INT32_MAX) || text.size() >= size_t(INT32_MAX) - length) {
            throw std::runtime_error(
                    "NanoQuant: Parquet calibration text exceeds INT32_MAX bytes");
        }
        text.append(value.data(), length);
        text.push_back('\n');
    }
}

static void append_parquet_array(
        const std::shared_ptr<arrow::Array> & array,
        std::string & text) {
    switch (array->type_id()) {
        case arrow::Type::STRING:
            append_parquet_strings(
                    *std::static_pointer_cast<arrow::StringArray>(array), text);
            return;
        case arrow::Type::LARGE_STRING:
            append_parquet_strings(
                    *std::static_pointer_cast<arrow::LargeStringArray>(array), text);
            return;
        case arrow::Type::BINARY:
            append_parquet_strings(
                    *std::static_pointer_cast<arrow::BinaryArray>(array), text);
            return;
        case arrow::Type::LARGE_BINARY:
            append_parquet_strings(
                    *std::static_pointer_cast<arrow::LargeBinaryArray>(array), text);
            return;
        default:
            throw std::runtime_error(
                    "NanoQuant: unsupported Parquet calibration column type '" +
                    array->type()->ToString() + "'");
    }
}

std::string llama_parquet_load_text(
        const char * path,
        const char * requested_column) {
    auto file_result = arrow::io::ReadableFile::Open(path);
    if (!file_result.ok()) {
        throw std::runtime_error(
                "NanoQuant: cannot open Parquet calibration dataset '" +
                std::string(path) + "': " + file_result.status().ToString());
    }
    std::shared_ptr<arrow::io::ReadableFile> file =
            std::move(file_result).ValueOrDie();
    auto reader_result = parquet::arrow::OpenFile(file, arrow::default_memory_pool());
    if (!reader_result.ok()) {
        throw std::runtime_error(
                "NanoQuant: cannot read Parquet calibration dataset '" +
                std::string(path) + "': " + reader_result.status().ToString());
    }
    std::unique_ptr<parquet::arrow::FileReader> reader =
            std::move(reader_result).ValueOrDie();
    std::shared_ptr<arrow::Schema> schema;
    const arrow::Status schema_status = reader->GetSchema(&schema);
    if (!schema_status.ok()) {
        throw std::runtime_error(
                "NanoQuant: cannot read Parquet schema from '" +
                std::string(path) + "': " + schema_status.ToString());
    }

    int column = -1;
    if (requested_column != nullptr && requested_column[0] != '\0') {
        column = schema->GetFieldIndex(requested_column);
        if (column < 0) {
            throw std::runtime_error(
                    "NanoQuant: Parquet calibration column '" +
                    std::string(requested_column) + "' does not exist");
        }
    } else {
        static constexpr const char * candidates[] = { "text", "content", "prompt" };
        for (const char * candidate : candidates) {
            for (int i = 0; i < schema->num_fields(); ++i) {
                std::string name = schema->field(i)->name();
                std::transform(name.begin(), name.end(), name.begin(),
                        [](unsigned char value) { return char(std::tolower(value)); });
                if (name == candidate && is_parquet_text_type(schema->field(i)->type()->id())) {
                    column = i;
                    break;
                }
            }
            if (column >= 0) {
                break;
            }
        }
        if (column < 0) {
            for (int i = 0; i < schema->num_fields(); ++i) {
                if (is_parquet_text_type(schema->field(i)->type()->id())) {
                    column = i;
                    break;
                }
            }
        }
    }
    if (column < 0) {
        throw std::runtime_error(
                "NanoQuant: Parquet dataset has no string or binary calibration column");
    }
    if (!is_parquet_text_type(schema->field(column)->type()->id())) {
        throw std::runtime_error(
                "NanoQuant: Parquet calibration column '" +
                schema->field(column)->name() + "' has unsupported type '" +
                schema->field(column)->type()->ToString() + "'");
    }

    LLAMA_LOG_INFO("NanoQuant: reading Parquet calibration column '%s'\n",
            schema->field(column)->name().c_str());
    std::string text;
    for (int row_group = 0; row_group < reader->num_row_groups(); ++row_group) {
        auto table_result = reader->ReadRowGroup(row_group, { column });
        if (!table_result.ok()) {
            throw std::runtime_error(
                    "NanoQuant: cannot read Parquet row group " +
                    std::to_string(row_group) + " from '" + path + "': " +
                    table_result.status().ToString());
        }
        const std::shared_ptr<arrow::Table> table =
                std::move(table_result).ValueOrDie();
        for (const std::shared_ptr<arrow::Array> & chunk : table->column(0)->chunks()) {
            append_parquet_array(chunk, text);
        }
    }
    if (text.empty()) {
        throw std::runtime_error(
                "NanoQuant: Parquet calibration column contains no non-null text");
    }
    return text;
}

#else

std::string llama_parquet_load_text(const char *, const char *) {
    throw std::runtime_error(
            "NanoQuant: Parquet support is not built; configure with -DLLAMA_PARQUET=ON");
}

#endif
