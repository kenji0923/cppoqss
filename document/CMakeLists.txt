find_package(Doxygen)

if (DOXYGEN_FOUND)
  set (DOXYGEN_SOURCE_BROWSER YES)
  set (DOXYGEN_INLINE_SOURCES YES)
  set (EXTRACT_ALL YES)
  message(STATUS ${PROJECT_SOURCE_DIR})
  doxygen_add_docs(cppoqss_doc
    ${LIBRARY_DIRECTORIES} ${EXECUTABLE_FILES}
    COMMENT "Generate manual pages"
  )
endif()
