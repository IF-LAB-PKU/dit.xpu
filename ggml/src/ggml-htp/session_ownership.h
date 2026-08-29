#pragma once

#include <string>
#include <unordered_map>

struct HtpSessionOwner {
    int session = 0;
    int domain = 3;
};

class HtpSessionOwnership {
public:
    bool record(const std::string & name, HtpSessionOwner owner) {
        auto it = owners_.find(name);
        if (it != owners_.end() &&
            (it->second.session != owner.session || it->second.domain != owner.domain)) {
            return false;
        }
        owners_[name] = owner;
        return true;
    }

    bool has(const std::string & name) const {
        return owners_.find(name) != owners_.end();
    }

    HtpSessionOwner owner(const std::string & name) const {
        auto it = owners_.find(name);
        return it == owners_.end() ? HtpSessionOwner{} : it->second;
    }

    bool needs_transfer(const std::string & name, HtpSessionOwner destination) const {
        auto it = owners_.find(name);
        return it != owners_.end() &&
               (it->second.session != destination.session ||
                it->second.domain != destination.domain);
    }

private:
    std::unordered_map<std::string, HtpSessionOwner> owners_;
};
