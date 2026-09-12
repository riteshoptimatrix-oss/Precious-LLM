import pytest
from unittest.mock import AsyncMock, MagicMock
from app.website.processing.versioning import WebsiteVersionManager

@pytest.mark.asyncio
async def test_version_manager_activate_version():
    db = MagicMock()
    vm = WebsiteVersionManager(db)
    
    vm.version_repo.list_versions = AsyncMock(return_value=[
        {"website_version": "v1.0.0", "is_active": False}
    ])
    vm.page_repo.set_active_version = AsyncMock()
    vm.chunk_repo.set_active_version = AsyncMock()
    vm.version_repo.activate_version = AsyncMock()
    vm.page_repo.get_active_pages = AsyncMock(return_value=[{"url": "https://www.preciousedu.in/"}])
    vm.chunk_repo.get_active_chunks = AsyncMock(return_value=[{"content": "c1"}])
    
    summary = await vm.activate_version("v1.0.0")
    
    assert summary["website_version"] == "v1.0.0"
    assert summary["pages_activated"] == 1
    assert summary["chunks_activated"] == 1
    assert summary["status"] == "ACTIVATED"

@pytest.mark.asyncio
async def test_version_manager_activate_nonexistent():
    db = MagicMock()
    vm = WebsiteVersionManager(db)
    vm.version_repo.list_versions = AsyncMock(return_value=[])
    
    with pytest.raises(ValueError) as exc:
        await vm.activate_version("v99.0.0")
    assert "not found in registry" in str(exc.value)
