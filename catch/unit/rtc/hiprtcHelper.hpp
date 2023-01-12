#ifndef HIPRTC_HELPER_HPP
#define HIPRTC_HELPER_HPP

#include <hip/hiprtc.h>

// Header for File Desc
#if defined(_WIN32)
  typedef void* FileDesc;
#else
  typedef int FileDesc;
#endif

// Templated Helper function
template <typename T> inline T alignDown(T value, size_t alignment) {
  return (T)(value & ~(alignment - 1));
}

template <typename T> inline T* alignDown(T* value, size_t alignment) {
  return (T*)alignDown((intptr_t)value, alignment);
}

template <typename T> inline T alignUp(T value, size_t alignment) {
  return alignDown((T)(value + alignment - 1), alignment);
}

template <typename T> inline T* alignUp(T* value, size_t alignment) {
  return (T*)alignDown((intptr_t)(value + alignment - 1), alignment);
}

// Exported Functions
bool CommitBCToFile(char* executable, size_t exe_size, const std::string& bit_code_file);

bool TestModuleLoadData(void* cuOut);
bool TestModuleLoad2Data(void* cuOut);
bool TestCompileRDC(char** bit_code_pptr, size_t* bit_code_size_ptr,
                    const char* routine_ptr, std::string routine_name);
bool TestCompileProgram(const char* source_code, hiprtcProgram* prog_ptr, std::string prog_name,
                        std::string func_name);

bool OpenFileHandle(const char* fname, FileDesc* fd_ptr, size_t* sz_ptr);
bool CloseFileHandle(FileDesc fdesc);

bool MemoryMapFileDesc(FileDesc fdesc, size_t fsize, size_t foffset, const void** mmap_pptr);
bool MemoryUnmapFile(const void* mmap_ptr, size_t mmap_size);

bool GetMapPtr(const char* fname, FileDesc* fd_ptr, size_t* sz_ptr, const void** mmap_pptr);
bool DelMapPtr(const void* mmap_ptr, size_t mmap_size);

#endif /* HIPRTC_HELPER_HPP */
